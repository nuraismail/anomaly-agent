############# AnomalyAgent.py #############

# user-defined modules

from utils.string_utils import cleanup_output_dir, parse_test_metadata, text_to_dict
from utils.summary_utils import build_default_summary, compact_summary
from utils.TeeIO import tee_stdout
from utils.family_novelty import extract_test_signature, compact_catalog_text, rotation_guidance_text, novelty_check, family_rotation_check
from utils.plot_functions import plot_results

# library

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import file_paths
from glob import glob
from dotenv import load_dotenv
from pathlib import Path
from typing import Annotated
from scipy.stats import norm
import io
import os
import re
import gc
import json
import yaml
import sqlite3
import traceback
import time
import itertools
import arxiv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.types import RetryPolicy, Command
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.messages import RemoveMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool, ToolRuntime
from langchain_community.tools import DuckDuckGoSearchResults

# environmental variables

load_dotenv()

# agent class

class AnomalyAgent:
    """
    LangGraph-based agentic AI tool that devises, executes, and evaluates CMB anomaly tests.
    
    args:
        model (str): Name of LLM model (available on OpenRouter)
        thread_id (str, optional): Label for conversation thread (used to maintain persistence)
    """
    
    def __init__(self, model:str, thread_id:str="thread_1"):
        
        # initialise variables
        
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing. Set it in environment variables/.env file.")

        self.thread_id = thread_id

        self.chain = None
        self.previous_state = None
        self.state = None

        self.sim_maps_path = file_paths.sim_maps_path
        self.planck_map_path = file_paths.planck_map_path
        self.output_root = file_paths.output_dir

        self.checkpoint_db_path = file_paths.checkpoint_db_path
        self.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

        self.test_output_dir = Path(self.output_root) / (Path(self.checkpoint_db_path).stem / Path(self.thread_id))
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        with open(file_paths.test_config_dir) as f:
            self.test_config = json.load(f)

        with open(file_paths.plot_config_dir) as f:
            self.plot_config = json.load(f)

        # initialise checkpointer

        conn = sqlite3.connect(self.checkpoint_db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        self.checkpointer = SqliteSaver(conn)

        # agent code execution output

        self.python_env = {
            "analyze_map": None,
            "summarize_results": None,
            "test_description": None,
            "last_error": None,
            "last_sigma": None,
            "last_result": None,
        }

        # initialise models

        self.llm = self.make_llm()
        self.search_llm = self.llm.bind_tools([self.web_search, self.arxiv_search], parallel_tool_calls=False)
        self.implement_llm = self.llm.bind_tools(self.register_tool(), parallel_tool_calls=False)

        # tool nodes

        self.planner_tool_node = ToolNode([self.web_search, self.arxiv_search], messages_key="search_query")
        self.implement_tool_node = ToolNode(self.register_tool())
        self.summary_tool_node = ToolNode([self.web_search, self.arxiv_search], messages_key="search_query")

    # non-state dependent functions

    def make_llm(self):
        """
        Initialises LLM model from OpenRouter.

        returns:
            ChatOpenAI: model instance
        """
        return ChatOpenAI(
            model=self.model,
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            reasoning={
            "effort": "xhigh",
        })
    
    def to_python_types(self, value):
        """
        Converts numpy and other non-standard types to native Python types for JSON serialization.
        
        args:
            value (any): The value to convert.
        returns:
            The converted value.
        """
        if isinstance(value, np.generic):
            return value.item()
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {key: self.to_python_types(val) for key, val in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self.to_python_types(item) for item in value]
        else:
            return value

    def check_runtime(self, start_time:int | float, max_time:int | float) -> None:
        """
        Checks if the elapsed time since start_time exceeds max_time (in minutes). If so, raises a RuntimeError.
        
        args:
            start_time (int | float): The time when the test started.
            max_time (int | float): The maximum allowed time in minutes.
        raises: 
            RuntimeError: If the elapsed time exceeds max_time.
        returns:
            None: If the elapsed time is within the limit.
        """
        elapsed_minutes = (time.time() - start_time) / 60.0
        if elapsed_minutes > max_time:
            if max_time <= 1:
                raise RuntimeError(
                    f"Restarting test: exceeded {max_time * 60} seconds."
                )
            else:
                raise RuntimeError(
                    f"Restarting test: exceeded {max_time} minutes."
                )
        else:
            return None

    # graph state

    class State(MessagesState):
        code: Annotated[list[AnyMessage], add_messages]
        test_hypothesis: Annotated[list[AnyMessage], add_messages]
        test_type: Annotated[list[AnyMessage], add_messages]
        justification: Annotated[list[AnyMessage], add_messages]
        execution_output: Annotated[list[AnyMessage], add_messages]
        test_summary: Annotated[list[AnyMessage], add_messages]
        search_query: Annotated[list[AnyMessage], add_messages]
        search_results: Annotated[list[AnyMessage], add_messages]
        search_count: int
        tested_anomalies: list[str]
        results_log: list[dict]
        current_results: dict
        current_test_name: str
        current_test_description: str
        best_sigma: float
        node_retry: bool
        python_env: dict
    
    # state dependent functions

    def retrieve_state(self, state: State, key: str, max_entries:int = None) -> str:
        """
        Retrieves and concatenates the content of messages from the state for a given key.
        
        args:
            state (State): The current state of the agent.
            key (str): The state key from which to retrieve messages.
            max_entries (int, optional): The maximum number of recent messages to retrieve.
        returns:
            str: The concatenated content of the retrieved messages.
        """
        if max_entries is not None and max_entries > 0:
            values = state.get(key, [])[-max_entries:]
        else:
            values = state.get(key, [])

        if not values:
            return ""
        else:
            return "\n\n".join(msg.content for msg in values if getattr(msg, "content", ""))

    def next_saved_test_index(self, state: State, test_success:bool=False) -> int:
        """
        Determines the next test index for saving results, based on the number of previously tested anomalies in the state and whether the current test was successful.
        
        args:
            state (State): The current state of the agent.
            test_success (bool, optional): Whether the current test was successful.
        returns:
            int: The next test index for saving results.
        """
        if not test_success:
            return len(state.get("tested_anomalies", [])) + 1
        else:
            return len(state.get("tested_anomalies", []))
        
    def current_output_dir(self, state: State, test_success:bool=False) -> Path:
        """
        Determines the output directory for the current test, based on the test index and the test name held in the state.
        
        args:
            state (State): The current state of the agent.
            test_success (bool, optional): Whether the current test was successful.
        returns:
            Path: The output directory for the current test.
        """
        test_index = self.next_saved_test_index(state, test_success)
        label = state.get("current_test_name", f"test_{test_index}")
        clean_label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        path = self.test_output_dir / f"Test_{test_index:02d}_{clean_label}"
        path.mkdir(parents=True, exist_ok=True)
        
        return path

    # anomaly test code execution functions

    def iter_simulation_maps(self):
        """
        Generator that yields simulated CMB maps from the specified directory, handling both individual .npy files and multi-map .npy files.

        raises:
            FileNotFoundError: If no simulation files are found in the specified directory.
        yields:
            tuple: A tuple containing the file path and the corresponding CMB map.
        """
        paths = sorted(glob(str(self.sim_maps_path)))
        if not paths:
            raise FileNotFoundError(f"No simulation files matched: {self.sim_maps_path}")

        for path in paths:
            arr = np.load(path, allow_pickle=False, mmap_mode="r")
            if arr.ndim == 1:
                yield path, np.asarray(arr, dtype=float)
            else:
                for idx in range(arr.shape[0]):
                    yield f"{path}[{idx}]", np.asarray(arr[idx], dtype=float)

    def prepare_planck_data(self, target_nside: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads the Planck CMB map and mask, degrades them to the target nside, and applies the mask to the map.
        
        args:
            target_nside (int): The nside to which the Planck map and mask should be degraded.
        returns:
            tuple: A tuple containing the masked Planck map and the mask itself.
        """
        planck_map_full = hp.read_map(self.planck_map_path, field=0)
        planck_mask_full = hp.read_map(self.planck_map_path, field=3)

        planck_map = hp.ud_grade(planck_map_full, target_nside) * 1e6
        planck_mask = hp.ud_grade(planck_mask_full, target_nside)
        planck_mask = planck_mask.astype(bool)

        planck_map_masked = np.asarray(planck_map, dtype=float).copy()
        planck_map_masked[~planck_mask] = np.nan
        
        return planck_map_masked, planck_mask

    def run_registered_analysis(self, state: State) -> str:
        """
        Executes the analyze_map(m) function on the Planck map and the simulation maps, computes the statistics, generates plots, and saves the results.
        
        args:
            state (State): The current state of the agent.
        returns:
            str: A message indicating the outcome/output of the analysis.
        """

        local_env = {}
        safe_globals = {"np": np, "hp": hp}

        try:
            test_codes = state.get("code", [None])
            if not test_codes or test_codes[-1].content is None:
                self.python_env["last_error"] = "No code found in state to execute."
                return "ERROR\n\nNo code found in state to execute."
            exec(test_codes[-1].content, safe_globals, local_env)
        except Exception:
            self.python_env["last_error"] = traceback.format_exc()
            return f"ERROR\n\n{self.python_env['last_error']}"

        if "analyze_map" not in local_env:
            self.python_env["last_error"] = "No registered analyze_map(m) function found."
            return "ERROR\n\nNo registered analyze_map(m) function found."
        else:
            stat_fn = local_env["analyze_map"]
            summary_fn = local_env.get("summarize_results")

        output_dir = self.current_output_dir(state)
        saved_test_index = self.next_saved_test_index(state)
        log = io.StringIO()
        self.python_env["last_result"] = None
        self.python_env["last_sigma"] = None

        sim_iter = self.iter_simulation_maps()

        try:
            first_path, first_map = next(sim_iter)
        except (StopIteration, FileNotFoundError):
            self.python_env["last_error"] = "No simulation maps were found."
            cleanup_output_dir(output_dir)
            return "ERROR\n\nNo simulation maps were found."

        try:
            target_nside = hp.get_nside(np.asarray(first_map))
            planck_map_masked, planck_mask = self.prepare_planck_data(target_nside)
            sim_results = []

            with tee_stdout(log):
                print(f"Test name: {state['current_test_name']}")
                print(f"Saved test index: {saved_test_index}")
                print(f"Output directory: {output_dir}")
                print(f"Using target nside={target_nside}")
                print(f"First simulation source: {first_path}")

                planck_stat = float(stat_fn(planck_map_masked))

                print(f"Observed Planck statistic: {planck_stat}")

                if not np.isfinite(planck_stat):
                    self.python_env["last_error"] = (
                        "Planck statistic is not finite. The registered analyze_map(m) likely "
                        "passed np.nan values into a routine that cannot handle masked pixels "
                        "directly. Update analyze_map(m) to ignore masked pixels with np.isfinite "
                        "and, for healpy transforms such as anafast/map2alm, replace masked np.nan "
                        "pixels with a finite fill value such as 0.0 before calling healpy."
                    )
                    print(f"ERROR: {self.python_env['last_error']}")
                    cleanup_output_dir(output_dir)
                    return "ERROR\n\n" + log.getvalue() + "\n" + self.python_env["last_error"]

                processed = 0
                attempt_start = time.time()

                for sim_source, sim_map in itertools.chain([(first_path, first_map)], sim_iter):
                    self.check_runtime(attempt_start, self.test_config["max_test_minutes"])
                    sim_map = np.asarray(sim_map, dtype=float)
                    sim_map_masked = sim_map.copy()
                    sim_map_masked[~planck_mask] = np.nan
                    sim_stat = float(stat_fn(sim_map_masked))
                    if not np.isfinite(sim_stat):
                        self.python_env["last_error"] = (
                            f"Simulation statistic became non-finite for {sim_source}. "
                            "The registered analyze_map(m) is not robust to masked pixels or "
                            "numerical edge cases."
                        )
                        print(f"ERROR: {self.python_env['last_error']}")
                        cleanup_output_dir(output_dir)
                        return "ERROR\n\n" + log.getvalue() + "\n" + self.python_env["last_error"]
                    sim_results.append(sim_stat)
                    processed += 1
                    if processed <= 10 or (processed <= 100 and processed % 10 == 0) or (processed <= 300 and processed % 50 == 0) or (processed <= 1000 and processed % 100 == 0):
                        print(f"Processed {processed} simulations (latest source: {sim_source})")

        except Exception:
            self.python_env["last_error"] = traceback.format_exc()
            self.python_env["last_sigma"] = None
            self.python_env["last_result"] = None
            self.python_env["analyze_map"] = None
            self.python_env["summarize_results"] = None
            cleanup_output_dir(output_dir)
            print(self.python_env["last_error"])
            return "ERROR\n\n" + log.getvalue() + "\n" + self.python_env["last_error"]
        finally:
            gc.collect()

        sim_results = np.asarray(sim_results, dtype=float)
        exceed_count = int(len(sim_results[sim_results >= planck_stat]))
        below_count = int(len(sim_results[sim_results <= planck_stat]))
        n_sims = int(sim_results.size)
        test_type = state["test_type"][-1].content.lower()
        p_upper = (exceed_count + 1) / (n_sims + 1)
        p_lower = (below_count + 1) / (n_sims + 1)
        
        if "two" in test_type:
            p_value = min(1.0, 2 * min(p_upper, p_lower))
            sigma = float(norm.isf(p_value / 2.0))
        elif "one" in test_type:
            if "upper" in test_type:
                p_value = p_upper
            elif "lower" in test_type:
                p_value = p_lower
            else:
                raise ValueError(f"One-tailed test missing upper/lower direction: {test_type}")
            sigma = float(norm.isf(p_value))
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        extra_summary = None
        raw_summary_output = None
        summary_source = "fallback_missing"
        summary_error = None

        if callable(summary_fn):
            try:
                extra_summary = summary_fn(planck_stat, sim_results)
                raw_summary_output = extra_summary
                summary_source = "model"
            except Exception:
                summary_error = traceback.format_exc()
                print("WARNING: summarize_results failed; using fallback summary.")
                summary_source = "fallback_error"

        if isinstance(extra_summary, dict):
            merged_summary = build_default_summary(planck_stat, sim_results, source=summary_source)
            merged_summary.update(self.to_python_types(extra_summary))
            extra_summary = merged_summary
        else:
            if extra_summary is not None and summary_source == "model":
                summary_source = "fallback_non_dict"
            extra_summary = build_default_summary(planck_stat, sim_results, source=summary_source)
            if summary_error is not None:
                extra_summary["summary_error"] = summary_error
            if summary_source == "fallback_non_dict":
                extra_summary["raw_summary_output"] = self.to_python_types(raw_summary_output)

        extra_summary = compact_summary(extra_summary)

        np.save(output_dir / "simulation_statistics.npy", sim_results)
        np.save(output_dir / "planck_statistic.npy", np.asarray([planck_stat]))
        plot_path = plot_results(
            planck_stat,
            sim_results,
            output_dir,
            summary=extra_summary if isinstance(extra_summary, dict) else None,
            plot_config=self.plot_config
        )

        result_payload = {
            "saved_test_index": saved_test_index,
            "test_name": state["current_test_name"],
            "test_description": state["current_test_description"],
            "test_hypothesis": state["test_hypothesis"][-1].content,
            "justification": state["justification"][-1].content,
            "planck_stat": planck_stat,
            "mean_sim_stat": float(np.mean(sim_results)),
            "std_sim_stat": float(np.std(sim_results)),
            "median_sim_stat": float(np.median(sim_results)),
            "p_value": float(p_value),
            "tail": state["test_type"][-1].content,
            "sigma": sigma,
            "n_sims": n_sims,
            "plot_path": plot_path["png"],
            "plot_pdf_path": plot_path["pdf"],
            "plot_kind": plot_path["kind"],
            "output_dir": str(output_dir),
            "custom_summary": self.to_python_types(extra_summary),
            "analysis_code": state["code"][-1].content if state.get("code") else None,
            "test_signature": extract_test_signature(state["current_test_name"], state["current_test_description"]),
        }

        self.python_env["last_error"] = None
        self.python_env["last_sigma"] = sigma
        self.python_env["last_result"] = result_payload

        return "CODE OUTPUT\n\n" + log.getvalue()
    
    def register(self, code: str, runtime:ToolRuntime) -> Command:
        """
        Registers Python code for CMB map analysis.
        
        args:
            code (str): The Python code to register, which must define an analyze_map(m) function.
            runtime (ToolRuntime): The runtime context for the tool execution.
        returns:
            Command: A command to update the agent's state with the registration result.
        """

        id = runtime.state["messages"][-1].tool_calls[0]['id']
        local_env = {}
        safe_globals = {"np": np, "hp": hp}

        try:
            exec(code, safe_globals, local_env)
        except Exception:
            self.python_env["last_error"] = traceback.format_exc()
            self.python_env["analyze_map"] = None
            self.python_env["summarize_results"] = None
            self.python_env["test_description"] = None
            output_msg = f"ERROR\n\n{self.python_env['last_error']}"
            return Command(update={"messages": [ToolMessage(output_msg, tool_call_id=id)], "node_retry": True, "python_env": self.python_env.copy()})

        if "analyze_map" not in local_env:
            self.python_env["last_error"] = "analyze_map(m) was not defined."
            self.python_env["analyze_map"] = None
            self.python_env["summarize_results"] = None
            self.python_env["test_description"] = None
            output_msg = "ERROR\n\nanalyze_map(m) was not defined."
            return Command(update={"messages": [ToolMessage(output_msg, tool_call_id=id)], "node_retry": True, "python_env": self.python_env.copy()})

        analyze_fn = local_env["analyze_map"]
        summary_fn = local_env.get("summarize_results")

        # Lightweight guardrails before the expensive full run.
        suspicious_patterns = [
            "ks_2samp",
            "return (",
            "return[",
            "query_disc(",
            "for i in range(npix)",
            "for pix in range(",
        ]
        matched = [pattern for pattern in suspicious_patterns if pattern in code]

        if matched:
            self.python_env["last_error"] = (
                "Rejected analysis because it appears to violate the single-scalar / efficient-statistic rule. "
                f"Matched patterns: {matched}"
            )
            self.python_env["analyze_map"] = None
            self.python_env["summarize_results"] = None
            self.python_env["test_description"] = None
            output_msg = f"ERROR\n\n{self.python_env['last_error']}"
            return Command(update={"messages": [ToolMessage(output_msg, tool_call_id=id)], "node_retry": True, "python_env": self.python_env.copy()})

        try:
            sim_iter = self.iter_simulation_maps()
            sample_items = []
            for _ in range(3):
                sample_items.append(next(sim_iter))

            first_source, first_map = sample_items[0]
            target_nside = hp.get_nside(np.asarray(first_map))
            planck_map_masked, planck_mask = self.prepare_planck_data(target_nside)

            attempt_start = time.time()
            planck_probe = analyze_fn(planck_map_masked)
            probe_max_time = self.test_config["max_test_minutes"] / 750
            self.check_runtime(attempt_start, probe_max_time)

            if not np.isscalar(planck_probe):
                raise TypeError(f"analyze_map(m) must return a scalar float, got {type(planck_probe).__name__}")
            
            planck_probe = float(planck_probe)

            if not np.isfinite(planck_probe):
                raise ValueError("analyze_map(m) returned a non-finite Planck statistic.")

            sim_probes = []

            for source, sim_map in sample_items:
                sim_map = np.asarray(sim_map, dtype=float)
                sim_masked = sim_map.copy()
                sim_masked[~planck_mask] = np.nan
                probe = analyze_fn(sim_masked)

                if not np.isscalar(probe):
                    raise TypeError(f"analyze_map(m) must return a scalar float, got {type(probe).__name__}")
                
                probe = float(probe)

                if not np.isfinite(probe):
                    raise ValueError(f"Simulation probe became non-finite for {source}.")
                
                sim_probes.append(probe)

            if planck_probe == 0.0 and all(probe == 0.0 for probe in sim_probes):
                raise ValueError(
                    "Degenerate statistic detected: Planck and the first few simulations all produced 0.0. "
                    "Redesign the analysis so it yields a meaningful non-trivial scalar."
                )

        except RuntimeError as e:
            if "Restarting test" in str(e):
                print(f"ERROR\n\nTime to process Planck map exceeded {probe_max_time * 60} seconds. The time to process the batch of simulated maps may exceed {self.test_config["max_test_minutes"]} minutes as a result.")
                raise
            else:
                raise
        except StopIteration:
            self.python_env["last_error"] = "Unable to preflight analysis because fewer than 3 simulation maps were available."
            self.python_env["analyze_map"] = None
            self.python_env["summarize_results"] = None
            self.python_env["test_description"] = None
            output_msg = f"ERROR\n\n{self.python_env['last_error']}"
            return Command(update={"messages": [ToolMessage(output_msg, tool_call_id=id)], "node_retry": True, "python_env": self.python_env.copy()})
        except Exception:
            self.python_env["last_error"] = traceback.format_exc()
            self.python_env["analyze_map"] = None
            self.python_env["summarize_results"] = None
            self.python_env["test_description"] = None
            output_msg = f"ERROR\n\nPreflight validation failed.\n\n{self.python_env['last_error']}"
            return Command(update={"messages": [ToolMessage(output_msg, tool_call_id=id)], "node_retry": True, "python_env": self.python_env.copy()})

        self.python_env["analyze_map"] = analyze_fn
        self.python_env["summarize_results"] = summary_fn
        self.python_env["test_description"] = local_env.get("test_description", "No description provided.")
        self.python_env["last_error"] = None
        output_msg = "SUCCESS\n\nAnalysis registered."
        return Command(update={"messages": [ToolMessage(output_msg, tool_call_id=id)], "node_retry": False})

    # tools

    def register_tool(self):
        """
        Wraps the register function as a LangChain tool.
        
        returns:
             list: A list containing the tool function.
        """
        @tool
        def register_analysis(code: str, runtime:ToolRuntime):
            """
            Tool function to register the map analysis code provided by the LLM.
            
            args:
                code (str): The Python code to register, which must define an analyze_map(m) function.
            returns:
                Command: A command to update the agent's state with the registration result.
            """
            return self.register(code, runtime)
        
        return [register_analysis]

    @tool
    def web_search(query: str, runtime: ToolRuntime) -> Command:
        """
        Searches the web (using DuckDuckGo) for a cosmology or CMB-analysis concept.
        
        args:
            query (str): The search query string.
        returns:
            Command: A command to update the agent's state with the search results.
        """

        results = DuckDuckGoSearchResults(num_results=5).run(query)
        id = runtime.state["search_query"][-1].tool_calls[0]['id']
        search_count = runtime.state["search_count"]

        return Command(update={"search_results": ['Query: ' + query + '\n\n' + 'Results:\n\n' + results + '\n'], "search_query": [ToolMessage("Success", tool_call_id=id)], "search_count": search_count + 1})

    @tool
    def arxiv_search(query: str, runtime: ToolRuntime) -> Command:
        """
        Search arXiv for a cosmology or CMB-analysis concept.

        args:
            query (str): The search query string.
        returns:
            Command: A command to update the agent's state with the search results.
        """

        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=3)
        results = list(client.results(search))
        results_str = '\n\n'.join(['\n'.join(['Title: ' + r.title, 'Published: ' + r.published.strftime('%Y/%m/%d'), 'Authors: ' + ', '.join([r.authors[i].name for i in range(len(r.authors) if len(r.authors) <= 5 else 5)]), 'Abstract: ' + r.summary]) for r in results])
        id = runtime.state["search_query"][-1].tool_calls[0]['id']
        search_count = runtime.state["search_count"]

        return Command(update={"search_results": ['Query: ' + query + '\n\n' + 'Results:\n\n' + results_str + '\n'], "search_query": [ToolMessage("Success", tool_call_id=id)], "search_count": search_count + 1})

    # nodes

    def planner_node(self, state: State):
        tested = state.get("tested_anomalies", [])
        tested = tested if tested else 'None'
        search_count = state.get("search_count", 0)
        prior_catalog_text = compact_catalog_text(self.test_output_dir)
        rotation_guidance = rotation_guidance_text(self.test_output_dir, self.test_config)
        planner_feedback = state["messages"][-1].content if "REJECTED" in state["messages"][-1].content else ""
        search_results = self.retrieve_state(state, "search_results", max_entries=3)
        msg = None

        if search_count >= self.test_config["max_searches_per_test"]:
            search_instruction = "- Do NOT call any search tools."
        else:
            search_instruction = "- You MUST call web_search or arxiv_search. Make only ONE tool call at a time."

        with open(file_paths.planner_dir) as stream:
            file = yaml.safe_load(stream)
            template = file["template"]
        
        prompt = PromptTemplate.from_template(template)
        prompt = prompt.format_prompt(search_instruction=search_instruction, tested=tested, prior_catalog_text=prior_catalog_text, rotation_guidance=rotation_guidance, search_count=search_count, search_results=search_results, planner_feedback=planner_feedback)

        print('\n##### PROMPT #####\n')
        print(prompt.to_string())

        msg = self.search_llm.invoke(prompt)

        if getattr(msg, "tool_calls", None):
            return {"search_query": [msg]}
        elif (not getattr(msg, "tool_calls", None) and (("tool_calls" in msg.content or "query:" in msg.content) or msg.content == '')) or getattr(msg, "invalid_tool_calls", None):
            return {"node_retry": True}
        else:
            if isinstance(msg.content, list) and any(d['type'] == 'text' for d in msg.content):
                idx = next((index for (index, d) in enumerate(msg.content) if d["type"] == "text"), None)
                test_name, test_description = parse_test_metadata(msg.content[idx]["text"])
            else:
                test_name, test_description = parse_test_metadata(msg.content)
            rotation_issue = family_rotation_check(self.test_output_dir, self.test_config, test_name, test_description)
            novelty_issue = novelty_check(self.test_output_dir, test_name, test_description)
            feedback_parts = []

        if rotation_issue is None and novelty_issue is None:
            return {
                "messages": [msg],
                "current_test_name": test_name,
                "current_test_description": test_description,
                "search_count": 0,
                "node_retry": False
            }
        elif rotation_issue is not None:
            if rotation_issue["severity"] == "blocked":
                feedback_parts.append(
                    "REJECTED FOR FAMILY OVERUSE:\n"
                    f"- {rotation_issue['message']}\n"
                    "- Choose a different anomaly family for the next proposal."
                )
            else:
                feedback_parts.append(
                    "REJECTED FOR FAMILY REPETITION:\n"
                    f"- {rotation_issue['message']}\n"
                    "- Prefer a different family unless you can make the proposal clearly distinct."
                )
        else: # novelty_issue is not None
            feedback_parts.append(
                "REJECTED FOR REPETITION:\n"
                f"- Your last proposal was too similar to prior test '{novelty_issue['prior_name']}'.\n"
                f"- Prior family: {novelty_issue['family']}. Similarity score: {novelty_issue['score']:.2f}.\n"
                "- Propose a substantively different test, ideally from a different anomaly family and with a different statistic form.\n"
                "- Do not make a cosmetic rename of the same dipole/quadrupole/octupole cross-correlation or power-asymmetry template."
            )

        planner_feedback = "\n\n".join(feedback_parts)

        return {
            "messages": [AIMessage(content=planner_feedback)],
            "node_retry": True,
    }
    
    def implement_node(self, state: State):
        test_name = state['current_test_name']
        test_description = state['current_test_description']
        previous_error = self.python_env.get("last_error", None) or state.get("python_env", {}).get("last_error", None)

        if previous_error:
            previous_code = self.retrieve_state(state, "code", max_entries=1)
            previous_code = previous_code if previous_code else 'None'
        else:
            previous_code = ""

        with open(file_paths.implement_dir) as stream:
            file = yaml.safe_load(stream)
            template = file["template"]
            additional_template = file["additional_template"]

        if previous_error:
            guidance_prompt = PromptTemplate.from_template(additional_template)
            guidance_prompt = guidance_prompt.format_prompt(previous_error=previous_error).to_string()
        else:
            guidance_prompt = ""

        prompt = PromptTemplate.from_template(template)
        prompt = prompt.format_prompt(test_name=test_name, test_description=test_description, previous_code=previous_code, guidance=guidance_prompt)

        print('\n##### PROMPT #####\n')
        print(prompt.to_string())

        msg = self.implement_llm.invoke(prompt)

        if getattr(msg, "tool_calls", None):
            code_text = msg.tool_calls[0]["args"].get("code", "")
            return {"messages": [msg], "code": [AIMessage(content=code_text)], "node_retry": False}
        else:
            should_retry = getattr(msg, "invalid_tool_calls", None) or not getattr(msg, "tool_calls", None)
            return {"messages": [msg], "node_retry": bool(should_retry)}
    
    def hypothesis_node(self, state: State):
        test_name = state['current_test_name']
        test_description = state['current_test_description']
        code = self.retrieve_state(state, "code", max_entries=1)

        with open(file_paths.hypothesis_dir) as stream:
            file = yaml.safe_load(stream)
            template = file["template"]

        prompt = PromptTemplate.from_template(template)
        prompt = prompt.format_prompt(test_name=test_name, test_description=test_description, code=code)

        print('\n##### PROMPT #####\n')
        print(prompt.to_string())

        msg = self.llm.invoke(prompt)

        if isinstance(msg.content, list) and any(d['type'] == 'text' for d in msg.content):
            idx = next((index for (index, d) in enumerate(msg.content) if d["type"] == "text"), None)
            message = msg.content[idx]["text"]
        else:
            message = msg.content

        stop_markers = [
            "HYPOTHESIS",
            "TEST_TYPE",
            "JUSTIFICATION",
        ]
        test_dict = text_to_dict(message, stop_markers)

        return {"messages": [msg],
                "test_hypothesis": [test_dict["HYPOTHESIS"]],
                "test_type": [test_dict["TEST_TYPE"]],
                "justification": [test_dict["JUSTIFICATION"]]}
    
    def execute_node(self, state: State):
        output = self.run_registered_analysis(state)
        sigma = self.python_env.get("last_sigma") or 0.0
        result = self.to_python_types(self.python_env.get("last_result") or {})
        updated_results = list(state.get("results_log", []))
        if result:
            updated_results.append(result)

        return {
            "messages": [AIMessage(content=output)],
            "execution_output": [AIMessage(content=output)],
            "current_results": result,
            "results_log": updated_results,
            "best_sigma": max(float(state.get("best_sigma", 0.0)), float(sigma)),
            "node_retry": output.startswith("ERROR"),
            "search_results": "",
            "python_env": {},
        }

    def summary_node(self, state: State):
        test_name = state['current_test_name']
        test_description = state['current_test_description']
        test_hypothesis = state['test_hypothesis'][-1].content
        test_type = state['test_type'][-1].content
        
        result_payload = state["current_results"]
        entries_to_remove = ('plot_path', 'plot_pdf_path', 'output_dir', 'test_signature')

        for k in entries_to_remove:
            result_payload.pop(k, None)

        result_payload = json.dumps(result_payload, indent=2, default=str)
        search_results = self.retrieve_state(state, "search_results", max_entries=1)

        with open(file_paths.summary_dir) as stream:
            file = yaml.safe_load(stream)
            template = file["template"]

            if state.get("search_count", 0) > self.test_config["max_searches_per_test"]:
                search_instruction = file["search_instruction_2"]
                search_instruction = PromptTemplate.from_template(search_instruction)

            else:
                search_instruction = file["search_instruction_1"]
                search_instruction = PromptTemplate.from_template(search_instruction)
            
            search_instruction = search_instruction.format_prompt(search_results=search_results).to_string()

        prompt = PromptTemplate.from_template(template)
        prompt = prompt.format_prompt(test_name=test_name, test_description=test_description, test_hypothesis=test_hypothesis, test_type=test_type, result_payload=result_payload, search_instruction=search_instruction)

        print('\n##### PROMPT #####\n')
        print(prompt.to_string())

        msg = self.search_llm.invoke(prompt)

        if getattr(msg, "tool_calls", None):
            return {"search_query": [msg], "search_count": state.get("search_count", 0) + 1}
        elif (not getattr(msg, "tool_calls", None) and (("tool_calls" in msg.content) or msg.content == '')) or getattr(msg, "invalid_tool_calls", None):
            return {"node_retry": True}
        else:
            if any(d['type'] == 'text' for d in msg.content):
                idx = next((index for (index, d) in enumerate(msg.content) if d["type"] == "text"), None)
                summary = msg.content[idx]["text"]
            else:
                summary = msg.content

            tested = list(state.get("tested_anomalies", []))
            tested.append(state["current_test_name"])

            return {
                "messages": [msg],
                "test_summary": summary,
                "tested_anomalies": tested,
                "search_count": 0,
                "node_retry": False,
            }
    
    # routing

    def planner_route(self, state: State):
        last = state.get("search_query", [])
        if state.get("node_retry") == True:
            print("Did not obtain correct output. Retrying...\n\n")
            return "planner"
        elif len(last) > 0 and getattr(last[-1], "tool_calls", False):
            return "planner_tools"
        else:
            return "implement"
        
    def implement_route(self, state: State):
        if getattr(state["messages"][-1], "tool_calls", False):
            return "implement_tools"
        else:
            print("Did not obtain correct output. Retrying...\n\n")
            return "implement"
        
    def post_register_route(self, state: State):
        if self.python_env.get("last_error", None):
            return "implement"
        else:
            return "hypothesis"

    def execute_route(self, state: State):
        if state.get("node_retry") == True:
            return "implement"
        else:
            return "summary"

    def summary_route(self, state: State):
        last = state.get("search_query", [])
        if len(last) > 0 and getattr(last[-1], "tool_calls", None):
            return "summary_tools"
        elif state.get("node_retry") == True:
            print("Did not obtain correct output. Retrying...\n\n")
            return "summary"
        else:
            out_dir = self.current_output_dir(state, test_success=True) / 'result_summary.json'
            data = state.get("current_results", {})
            stop_markers = [
                    "Test name",
                    "Description",
                    "Results",
                    "Interpretation",
                    "Meets hypothesis?",
                    "Comparison to literature",
                    "Test novelty",
                    "Verdict",
                ]
            data["test_summary"] = text_to_dict(state["test_summary"][-1].content, stop_markers)
            with open(out_dir, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            return "__end__"

    # main loop

    def run(self):
        directories = [x[0] for x in os.walk(self.test_output_dir)]

        tested_anomalies = []

        for dir in directories[1:]:
            out_dir = Path(dir) / 'result_summary.json'
            try:
                with open(out_dir, 'r+') as f:
                    data = json.load(f)
                    summary = data.get("test_summary", None)
                    if summary != None:
                        tested_anomalies.append(data['test_summary']['Test name'])
            except FileNotFoundError:
                continue

        # Initialise config - thread_id needs to be the same so that the state can persist across multiple runs
        run_config = {
                "configurable": {
                    "thread_id": f"{self.thread_id}"
                }
            }
        
        initial_input = {
            "messages": "",
            "search_count": 0,
            "tested_anomalies": tested_anomalies,
            "current_results": "",
            "current_test_name": "",
            "current_test_description": "",
            "node_retry": False,
            "search_results": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]
        }

        state = self.chain.get_state(run_config)
        if state.next:
            print("Resuming from checkpoint...\n\n")
            pending_input = None
        else:
            pending_input = initial_input

        test_count = len(tested_anomalies)
        tests_remaining = self.test_config['tests_to_run'] - test_count

        for test_idx in range(tests_remaining):
            restart_attempt = 0
            test_completed = False

            while restart_attempt <= self.test_config["max_test_restarts"] and not test_completed:
                restart_test = False
                print(f"\n===== RUNNING TEST {(test_count) + 1} / {self.test_config['tests_to_run']} | ATTEMPT {restart_attempt + 1} =====\n")

                retry_count = 3
                attempt_start = time.time()
                implement_visits = 0
                execute_errors = 0

                while not test_completed:
                    try:
                        for event in self.chain.stream(pending_input, config=run_config):
                            elapsed_minutes = (time.time() - attempt_start) / 60.0

                            for node, output in event.items():
                                if not output:
                                    continue

                                print(f"\n===== {node.upper()} =====")

                                if "search_query" in output and output["search_query"]:
                                    msg = output["search_query"][-1]
                                elif "search_results" in output and output["search_results"]:
                                    msg = output["search_results"][-1]
                                elif "messages" in output and output["messages"]:
                                    msg = output["messages"][-1]
                                else:
                                    msg = output

                                if getattr(msg, "content", None):
                                    if isinstance(msg.content, str):
                                        print("CONTENT:" + '\n\n' + msg.content[:2000] + '\n\n')
                                    elif isinstance(msg.content, list) and any(isinstance(d, dict) for d in msg.content):
                                        if any(d['type'] == 'reasoning' for d in msg.content):
                                            summary_idx = next((index for (index, d) in enumerate(msg.content) if d["type"] == "reasoning"), None)
                                            if len(msg.content[summary_idx]["summary"]) > 0:
                                                print("REASONING:" + '\n\n' + str(msg.content[0]["summary"][0]["text"]) + '\n\n')
                                            else:
                                                print("REASONING:" + '\n\n' + str(msg.content[0]["summary"]) + '\n\n')
                                        if any(d['type'] == 'text' for d in msg.content):
                                            text_idx = next((index for (index, d) in enumerate(msg.content) if d["type"] == "text"), None)
                                            print("CONTENT:" + '\n\n' + msg.content[text_idx]["text"] + '\n\n')
                                    else:
                                        print("CONTENT:" + '\n\n' + msg.content + '\n\n')
                                else:
                                    print("CONTENT:" + '\n\n' + msg + '\n\n')

                                if getattr(msg, "tool_calls", None):
                                    print("TOOL CALLS:\n\n", msg.tool_calls)

                                if node == "implement":
                                    implement_visits += 1

                                if node == "execute" and isinstance(msg.content, str) and msg.content.startswith("ERROR"):
                                    execute_errors += 1

                            if elapsed_minutes > self.test_config["max_test_minutes"] and ((node != "summary" and node != "summary_tools" and node != "implement" and node != "execute") or (node == "execute" and output["node_retry"] == True)):
                                raise RuntimeError(
                                    f"Restarting test {test_count + 1}: exceeded {self.test_config['max_test_minutes']} minutes."
                                )

                            if implement_visits > self.test_config["max_fix_cycles_per_test"]:
                                raise RuntimeError(
                                    f"Restarting test {test_count + 1}: exceeded {self.test_config['max_fix_cycles_per_test']} implement/fix cycles."
                                )

                            if execute_errors > self.test_config["max_execute_errors_per_test"]:
                                raise RuntimeError(
                                    f"Restarting test {test_count + 1}: exceeded {self.test_config['max_execute_errors_per_test']} execution errors."
                                )

                        self.state = self.chain.get_state(run_config).values
                        test_completed = True
                        pending_input = initial_input
                        initial_input["tested_anomalies"] = self.state.get("tested_anomalies", [])
                        restart_attempt = 0
                        test_count += 1

                    except RuntimeError as e:
                        print(str(e))
                        if "Restarting" in str(e):
                            restart_test = True
                            break
                        else:
                            raise

                    except ValueError as e:
                        error = str(e)
                        print(f"Transient or provider error: {error}")

                        is_transient = any(
                            code in error
                            for code in ["'code': 524", "'code': 500", "'code': 502", "'code': 504", '"code": 524', '"code": 500', '"code": 502', '"code": 504', "Expecting value:"]
                        )
                        if not is_transient or retry_count <= 0:
                            if "ToolMessage" in error:
                                restart_test = True
                                break
                            else:
                                raise

                        print(f"Retrying from checkpoint... attempts remaining: {retry_count}")
                        time.sleep(10)
                        pending_input = None
                        retry_count -= 1

                if restart_test == True:
                    restart_attempt += 1
                    pending_input = initial_input
                    if restart_attempt <= self.test_config["max_test_restarts"]:
                        print(
                            f"\nRestarting test {test_count + 1} from scratch with a fresh thread "
                            f"(attempt {restart_attempt + 1} of {self.test_config['max_test_restarts'] + 1}).\n"
                        )
                    else:
                        print(
                            f"\nSkipping test {test_count + 1} after {self.test_config['max_test_restarts'] + 1} failed attempts.\n"
                        )
                        test_count += 1

    # build graph and run

    def __call__(self):
        workflow = StateGraph(self.State)

        workflow.add_node("planner", self.planner_node, retry_policy=RetryPolicy(max_attempts=3, initial_interval=5, backoff_factor=2))
        workflow.add_node("planner_tools", self.planner_tool_node, retry_policy=RetryPolicy(max_attempts=5, initial_interval=5, backoff_factor=2))
        workflow.add_node("implement", self.implement_node, retry_policy=RetryPolicy(max_attempts=3, initial_interval=3))
        workflow.add_node("implement_tools", self.implement_tool_node)
        workflow.add_node("hypothesis", self.hypothesis_node)
        workflow.add_node("execute", self.execute_node, retry_policy=RetryPolicy(max_attempts=2, initial_interval=3))
        workflow.add_node("summary", self.summary_node, retry_policy=RetryPolicy(max_attempts=3, initial_interval=5, backoff_factor=2))
        workflow.add_node("summary_tools", self.summary_tool_node, retry_policy=RetryPolicy(max_attempts=5, initial_interval=5, backoff_factor=2))

        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self.planner_route,
            {"planner_tools": "planner_tools", "implement": "implement", "planner": "planner"}
            )
        workflow.add_edge("planner_tools", "planner")
        workflow.add_conditional_edges(
            "implement",
            self.implement_route,
            {"implement_tools": "implement_tools", "implement": "implement"},
        )
        workflow.add_conditional_edges("implement_tools", self.post_register_route, {"implement": "implement", "hypothesis": "hypothesis"})
        workflow.add_edge("hypothesis", "execute")
        workflow.add_conditional_edges(
            "execute",
            self.execute_route,
            {"implement": "implement", "summary": "summary"},
        )
        workflow.add_conditional_edges(
            "summary",
            self.summary_route,
            {"summary_tools": "summary_tools", "summary": "summary", "__end__": END}
        )
        workflow.add_edge("summary_tools", "summary")

        self.chain = workflow.compile(checkpointer=self.checkpointer)

        self.run()

if __name__ == "__main__":
    # model = "openai/gpt-5.5"
    # model = "openrouter/owl-alpha"
    model = "poolside/laguna-m.1:free"
    thread_id = "test_run_5"
    agent = AnomalyAgent(model, thread_id)
    agent()