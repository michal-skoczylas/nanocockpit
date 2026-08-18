"""
Test suite for NANOCKPIT_ARCHITECTURE.md documentation.

Verifies the architecture documentation file exists, is complete, and is
syntactically valid. Covers Task 2 (Architecture Overview) and Task 3
(Mermaid Flowcharts) from PLAN.md.
"""

import os
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCH_FILE = PROJECT_ROOT / "docs" / "NANOCKPIT_ARCHITECTURE.md"

# The four hardware/software nodes defined in Task 2
EXPECTED_NODES = ["STM32", "GAP8", "ESP32", "Host"]

# Key communication protocols expected in the architecture description
EXPECTED_PROTOCOLS = ["CPX", "UART", "SPI", "TCP", "UDP", "CRTP"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_arch_file():
    """Return the full text content of NANOCKPIT_ARCHITECTURE.md or None."""
    if not ARCH_FILE.exists():
        return None
    return ARCH_FILE.read_text(encoding="utf-8")


def _extract_mermaid_blocks(text: str):
    """
    Extract all fenced Mermaid code blocks from the markdown text.

    Returns a list of dicts with keys: 'start_line' (1-based), 'content'.
    """
    # Matches ```mermaid (maybe with lang alias) ... ```
    pattern = re.compile(
        r"^```mermaid\s*\n(.*?)^```",
        re.MULTILINE | re.DOTALL,
    )
    blocks = []
    for match in pattern.finditer(text):
        # Count newlines before the match to get 1-based start line
        line_number = text[: match.start()].count("\n") + 2  # +2: 0-based→1-based + skip the fence line
        blocks.append({
            "start_line": line_number,
            "content": match.group(1),
        })
    return blocks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def arch_text():
    """
    Module-scoped fixture returning the architecture doc contents.

    Skips the entire module if the file does not exist, so all tests
    that depend on it receive a clean skip rather than a confusing failure.
    """
    text = _read_arch_file()
    if text is None:
        pytest.skip(f"Architecture file not found: {ARCH_FILE}")
    return text


# ---------------------------------------------------------------------------
# Test 1 – File existence
# ---------------------------------------------------------------------------

def test_file_exists():
    """Verify that NANOCKPIT_ARCHITECTURE.md exists on disk."""
    assert ARCH_FILE.exists(), (
        f"Architecture documentation file not found at: {ARCH_FILE}\n"
        f"Expected location: docs/NANOCKPIT_ARCHITECTURE.md"
    )


# ---------------------------------------------------------------------------
# Test 2 – File size & minimum line count
# ---------------------------------------------------------------------------

def test_file_is_non_empty(arch_text):
    """Verify the file contains at least some content."""
    assert len(arch_text.strip()) > 0, "Architecture file is empty (no content)."


def test_minimum_line_count(arch_text):
    """Verify the documentation is detailed (at least 100 lines)."""
    lines = arch_text.splitlines()
    assert len(lines) >= 100, (
        f"Architecture documentation is too short: "
        f"expected >= 100 lines, got {len(lines)}. "
        f"Expand with more detail to reach the threshold."
    )


# ---------------------------------------------------------------------------
# Test 3 – All four hardware nodes are mentioned
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("node", EXPECTED_NODES)
def test_node_mentioned(arch_text, node):
    """
    Verify that each of the four system nodes is mentioned by name.

    This covers the Task 2 requirement: "Opis 4 węzłów systemu
    (STM32, GAP8, ESP32, Host PC)."
    """
    assert node.lower() in arch_text.lower(), (
        f"Node '{node}' is not mentioned in the architecture documentation. "
        f"All four nodes (STM32, GAP8, ESP32, Host) must be described."
    )


# ---------------------------------------------------------------------------
# Test 4 – Key communication protocols are mentioned
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("protocol", EXPECTED_PROTOCOLS)
def test_protocol_mentioned(arch_text, protocol):
    """
    Verify that every key communication protocol is mentioned.

    Protocols: CPX (packet protocol), UART, SPI (intra-drone buses),
    TCP, UDP (WiFi transport), CRTP (Crazyflie radio protocol).
    """
    assert protocol.upper() in arch_text.upper(), (
        f"Protocol '{protocol}' is not mentioned in the architecture "
        f"documentation. All key protocols (CPX, UART, SPI, TCP, UDP, "
        f"CRTP) should be described."
    )


# ---------------------------------------------------------------------------
# Test 5 – Mermaid diagrams present
# ---------------------------------------------------------------------------

def test_at_least_one_mermaid_diagram(arch_text):
    """
    Verify the file contains at least one Mermaid diagram definition.

    Task 3 requires Mermaid flowcharts for system communication.
    """
    blocks = _extract_mermaid_blocks(arch_text)
    assert len(blocks) >= 1, (
        "No Mermaid diagram found. The documentation must include at "
        "least one Mermaid code block (```mermaid ... ```)."
    )


# ---------------------------------------------------------------------------
# Test 6 – Mermaid syntax validation
# ---------------------------------------------------------------------------

# Valid Mermaid diagram type keywords
MERMAID_KEYWORDS = [
    "graph",
    "flowchart",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "erDiagram",
    "gantt",
    "pie",
    "quadrantChart",
    "requirementDiagram",
    "gitGraph",
    "mindmap",
    "timeline",
    "sankey",
    "block",
]


def test_no_unmatched_backticks(arch_text):
    """
    Sanity check: the file should not have unmatched triple-backtick fences.

    An odd number of ``` occurrences usually signals a broken Mermaid block.
    """
    backtick_triples = arch_text.count("```")
    assert backtick_triples % 2 == 0, (
        f"Unmatched triple-backtick fences detected: "
        f"found {backtick_triples} occurrence(s). "
        f"Every ```mermaid block must be properly closed with ```."
    )


def test_mermaid_blocks_have_valid_keyword(arch_text):
    """
    Verify every extracted Mermaid block starts with a valid diagram keyword.

    A Mermaid block that does not contain any known keyword (graph,
    flowchart, sequenceDiagram, etc.) is likely malformed.
    """
    blocks = _extract_mermaid_blocks(arch_text)
    assert len(blocks) > 0, (
        "No Mermaid blocks to validate. (This should have been caught by "
        "the previous test.)"
    )

    for block in blocks:
        content_stripped = block["content"].strip()
        found = any(kw in content_stripped for kw in MERMAID_KEYWORDS)
        assert found, (
            f"Mermaid block starting near line {block['start_line']} "
            f"does not contain any recognised diagram keyword "
            f"(e.g. 'graph', 'flowchart', 'sequenceDiagram'). "
            f"Block content starts with: "
            f"{content_stripped[:80]!r}"
        )


# ---------------------------------------------------------------------------
# Test 7 – Valid Markdown structure
# ---------------------------------------------------------------------------

def test_has_markdown_headers(arch_text):
    """
    Verify the file contains at least one Markdown heading (#).

    A documentation file without headers is not properly structured.
    """
    assert re.search(r"^#{1,6}\s+\S", arch_text, re.MULTILINE), (
        "No Markdown heading found (e.g., '# Title'). "
        "The documentation should use # headers for structure."
    )


def test_has_multiple_sections(arch_text):
    """
    Verify the document has at least 3 distinct headings, indicating
    reasonable section organisation.
    """
    headings = re.findall(r"^#{1,6}\s+\S.*$", arch_text, re.MULTILINE)
    assert len(headings) >= 3, (
        f"Only {len(headings)} heading(s) found; expected at least 3. "
        f"The architecture documentation should be split into clear "
        f"sections (e.g., Overview, Architecture, Communication Flow, "
        f"Diagrams)."
    )


def test_no_trailing_whitespace_on_significant_lines(arch_text):
    """
    Soft-style check: lines with content should not have trailing spaces.
    This is a best-practice markdown hygiene check; it does not fail hard
    but reports a warning.
    """
    violations = []
    for i, line in enumerate(arch_text.splitlines(), start=1):
        if line and line != line.rstrip():
            violations.append(i)

    if violations:
        pytest.fail(
            f"Trailing whitespace found on {len(violations)} line(s): "
            f"lines {violations[:10]}{'...' if len(violations) > 10 else ''}. "
            f"Please trim trailing spaces for clean Markdown."
        )


# ---------------------------------------------------------------------------
# Optional / smoke tests – run only when file exists
# ---------------------------------------------------------------------------

def test_file_encoding_is_utf8():
    """Verify the file is valid UTF-8."""
    if not ARCH_FILE.exists():
        pytest.skip("File not found – skipping encoding check.")
    try:
        ARCH_FILE.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"File is not valid UTF-8: {exc}")


def test_architecture_overview_section_present(arch_text):
    """
    Verify there is a section dedicated to the architecture overview
    (covering Task 2).
    """
    headings_text = "\n".join(
        re.findall(r"^#{1,6}\s+\S.*$", arch_text, re.MULTILINE)
    ).lower()
    relevant = any(
        term in headings_text
        for term in ["architecture", "architektura", "overview", "przegląd"]
    )
    assert relevant, (
        "No section heading related to 'Architecture' or 'Overview' found. "
        "The documentation should include a dedicated architecture section."
    )


# ---------------------------------------------------------------------------
# GAP8 Component Section Tests
# ---------------------------------------------------------------------------

# Example project names expected to be listed in the GAP8 section
GAP8_EXAMPLES = [
    "pulp-frontnet",
    "streamer",
    "cpx",
    "coroutine",
    "queue",
    "queue_async",
]

# Key library components / filenames expected in the GAP8 section
GAP8_LIB_COMPONENTS = [
    "camera",
    "coroutine",
    "queue",
    "streamer",
    "uart_protocol",
    "CPX",
    "cpx.h",
    "cluster",
    "crc32",
    "himax",
]

# Key configuration parameters for PULP-Frontnet
GAP8_PULP_FRONTNET_PARAMS = [
    "HIMAX_FORMAT",
    "HIMAX_FRAME_RATE",
    "SOC_VOLTAGE",
    "NETWORK_ONBOARD_INFERENCE",
    "BOARD",
    "CAMERA_BUFFERS",
]


def test_gap8_section_present(arch_text):
    """
    Verify that the documentation contains a dedicated section (heading)
    covering the GAP8 component in depth.
    """
    headings = re.findall(r"^#{1,6}\s+.*$", arch_text, re.MULTILINE)
    gap8_section = [
        h for h in headings if "gap8" in h.lower() or "GAP8" in h
    ]
    assert len(gap8_section) > 0, (
        "No dedicated GAP8 section/heading found in the architecture "
        "documentation. Expected at least one heading containing 'GAP8'."
    )


def test_gap8_sdk_version_mentioned(arch_text):
    """
    Verify that the GAP SDK version 3.8.1 is explicitly mentioned in the
    documentation.
    """
    assert "3.8.1" in arch_text, (
        "GAP SDK version '3.8.1' not found in the documentation. "
        "The GAP8 component section should mention the SDK version."
    )


@pytest.mark.parametrize("example_name", GAP8_EXAMPLES)
def test_gap8_examples_mentioned(arch_text, example_name):
    """
    Verify that each of the 6 GAP8 example projects is mentioned by name
    in the documentation.
    """
    assert example_name.lower() in arch_text.lower(), (
        f"GAP8 example '{example_name}' is not mentioned in the architecture "
        f"documentation. All 6 examples (pulp-frontnet, streamer, cpx, "
        f"coroutine, queue, queue_async) should be documented."
    )


@pytest.mark.parametrize("lib_name", GAP8_LIB_COMPONENTS)
def test_gap8_library_components_mentioned(arch_text, lib_name):
    """
    Verify that each key GAP8 library component is mentioned by name
    in the documentation.
    """
    # Some names are case-sensitive (e.g., 'CPX', 'himax'), so we use
    # a case-insensitive search for most but require exact for short tokens
    # like 'CPX' which could match too broadly if lowercased.
    if lib_name in ("CPX", "cpx.h"):
        assert lib_name in arch_text, (
            f"GAP8 library component '{lib_name}' not found in the "
            f"documentation (case-sensitive)."
        )
    else:
        assert lib_name.lower() in arch_text.lower(), (
            f"GAP8 library component '{lib_name}' is not mentioned in the "
            f"architecture documentation."
        )


def test_gap8_build_commands_mentioned(arch_text):
    """
    Verify that the GAP8 build/flash/run commands are documented.
    Accepts either 'make all flash' or 'make all run'.
    """
    has_flash = "make all flash" in arch_text
    has_run = "make all run" in arch_text
    assert has_flash or has_run, (
        "Neither 'make all flash' nor 'make all run' build command found "
        "in the documentation. The GAP8 section should describe how to "
        "build, flash, and run the firmware."
    )


def test_gap8_parameter_tables_present(arch_text):
    """
    Verify the document contains at least one Markdown table with
    configuration parameters (rows with | ... | ... | syntax).
    """
    table_row_pattern = re.compile(r"^\|.+\|.+\|", re.MULTILINE)
    matches = table_row_pattern.findall(arch_text)
    assert len(matches) >= 1, (
        "No Markdown parameter table found (rows matching '| ... | ... |'). "
        "The GAP8 section should include at least one table listing "
        "configuration parameters."
    )


@pytest.mark.parametrize("param_name", GAP8_PULP_FRONTNET_PARAMS)
def test_gap8_pulp_frontnet_params_section_present(arch_text, param_name):
    """
    Verify that key PULP-Frontnet configuration parameters are mentioned
    in the documentation.
    """
    assert param_name.upper() in arch_text.upper(), (
        f"PULP-Frontnet configuration parameter '{param_name}' not found in "
        f"the architecture documentation. The GAP8 section should document "
        f"all key Makefile/header configuration parameters."
    )


def test_gap8_minimum_total_lines(arch_text):
    """
    Verify that after adding the GAP8 section the documentation file
    has at least 400 lines (significant expansion from the original ~280).
    """
    lines = arch_text.splitlines()
    assert len(lines) >= 400, (
        f"Architecture documentation is too short: expected >= 400 lines "
        f"after adding the GAP8 section, but got {len(lines)}. "
        f"Expand the GAP8 component documentation."
    )


# ---------------------------------------------------------------------------
# STM32 Component Section Tests (Task 5)
# ---------------------------------------------------------------------------

# Source files from the STM32 PULP-Frontnet controller app (from Makefile)
STM32_SOURCE_FILES = [
    "frontnet_main",
    "frontnet_kf",
    "frontnet_ctrl",
    "frontnet_state_fwd",
    "frontnet_aideck_protocol",
    "frontnet_appchannel",
    "frontnet_rng",
    "frontnet_test_inferences",
]

# Compile-time parameters from frontnet_config.h
STM32_COMPILE_PARAMS = [
    "FRONTNET_TIMER_RATE",
    "STATE_FWD_RATE",
    "FRONTNET_INFERENCE_TIMEOUT",
    "FRONTNET_MIN_BATTERY_VOLTAGE",
    "FRONTNET_SETPOINT_PRIORITY",
]

# Runtime parameters from the CFclient 'frontnet' group (representative sample)
STM32_RUNTIME_PARAMS = [
    "frontnet.enable_control",
    "frontnet.kalman_x_r",
    "frontnet.distance",
    "frontnet.max_speed",
    "frontnet.min_voltage",
]

# Log group entries — 'frontnet' read-only telemetry (representative sample)
STM32_LOG_ENTRIES = [
    "control_enabled",
    "update_frequency",
    "inf_latency",
    "f_x",
    "f_vx",
]


# ---------------------------------------------------------------------------
# Test – STM32 dedicated section exists
# ---------------------------------------------------------------------------

def test_stm32_section_present(arch_text):
    """
    Verify that a dedicated section heading covering the STM32 component
    in depth exists in the documentation.

    Accepts a heading containing 'STM32' alongside any of: 'controller',
    'firmware', 'app', 'pulp-frontnet', 'component', 'section', 'detailed'.
    Falls back to accepting any heading with 'STM32' and 'component'.
    """
    headings = re.findall(r"^#{1,6}\s+.*$", arch_text, re.MULTILINE)

    # Primary match: STM32 + 'component' in same heading
    stm32_section = [
        h for h in headings
        if "stm32" in h.lower() and "component" in h.lower()
    ]
    # Secondary match: STM32 + any keyword suggesting in-depth coverage
    if not stm32_section:
        stm32_section = [
            h for h in headings
            if "stm32" in h.lower()
            and any(
                kw in h.lower()
                for kw in ["controller", "firmware", "app", "pulp-frontnet",
                           "section", "detailed"]
            )
        ]
    assert len(stm32_section) > 0, (
        "No dedicated STM32 section/heading found in the architecture "
        "documentation. Expected at least one heading containing 'STM32' "
        "that describes the component in depth (e.g., 'STM32 Component', "
        "'STM32 Firmware', 'STM32 PULP-Frontnet Controller')."
    )


# ---------------------------------------------------------------------------
# Test – Firmware version
# ---------------------------------------------------------------------------

def test_stm32_firmware_version_mentioned(arch_text):
    """
    Verify that the Crazyflie firmware fork version (v2021.06) is
    explicitly mentioned in the STM32 section.
    """
    assert "2021.06" in arch_text, (
        "Crazyflie firmware version '2021.06' not found in the documentation. "
        "The STM32 component section should mention the Bitcraze "
        "crazyflie-firmware v2021.06 fork."
    )


# ---------------------------------------------------------------------------
# Test – Build commands
# ---------------------------------------------------------------------------

def test_stm32_build_commands_mentioned(arch_text):
    """
    Verify that the STM32 build and flash commands are documented,
    specifically 'make cload'.
    """
    assert "make cload" in arch_text, (
        "'make cload' build command not found in the documentation. "
        "The STM32 section should describe how to build and flash the "
        "STM32 firmware: 'cd src/stm32/app && make clean all && make cload'."
    )


# ---------------------------------------------------------------------------
# Test – Source files (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source_file", STM32_SOURCE_FILES)
def test_stm32_source_files_mentioned(arch_text, source_file):
    """
    Verify that each of the 8 key STM32 source files is mentioned in the
    documentation.

    These files form the PULP-Frontnet controller app running on the STM32F405.
    """
    assert source_file.lower() in arch_text.lower(), (
        f"STM32 source file '{source_file}' is not mentioned in the "
        f"architecture documentation. All 8 source files "
        f"(frontnet_main.c, frontnet_kf.c, frontnet_ctrl.c, "
        f"frontnet_state_fwd.c, frontnet_aideck_protocol.c, "
        f"frontnet_appchannel.c, frontnet_rng.c, "
        f"frontnet_test_inferences.c) should be documented."
    )


# ---------------------------------------------------------------------------
# Test – Compile-time parameters (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param_name", STM32_COMPILE_PARAMS)
def test_stm32_compile_params_mentioned(arch_text, param_name):
    """
    Verify that key compile-time configuration parameters (from
    frontnet_config.h) are mentioned in the documentation.
    """
    assert param_name.upper() in arch_text.upper(), (
        f"STM32 compile-time parameter '{param_name}' not found in the "
        f"architecture documentation. The STM32 section should document "
        f"all key compile-time parameters from frontnet_config.h."
    )


# ---------------------------------------------------------------------------
# Test – Runtime parameters (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("param_name", STM32_RUNTIME_PARAMS)
def test_stm32_runtime_params_mentioned(arch_text, param_name):
    """
    Verify that key runtime-tunable parameters from the 'frontnet' CRTP
    parameter group are mentioned in the documentation.
    """
    assert param_name.lower() in arch_text.lower(), (
        f"STM32 runtime parameter '{param_name}' not found in the "
        f"architecture documentation. The STM32 section should document "
        f"the 21 runtime-tunable parameters in the 'frontnet' group."
    )


# ---------------------------------------------------------------------------
# Test – Log group entries (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("log_entry", STM32_LOG_ENTRIES)
def test_stm32_log_group_mentioned(arch_text, log_entry):
    """
    Verify that key log telemetry entries from the 'frontnet' read-only
    log group are mentioned in the documentation.
    """
    assert log_entry.lower() in arch_text.lower(), (
        f"STM32 log entry '{log_entry}' not found in the architecture "
        f"documentation. The STM32 section should document the 'frontnet' "
        f"log group with read-only telemetry variables."
    )


# ---------------------------------------------------------------------------
# Test – Default follow-me target distances
# ---------------------------------------------------------------------------

def test_stm32_default_target_mentioned(arch_text):
    """
    Verify that the default follow-me target distances are mentioned:
    horizontal distance = 1.5 m and altitude = 1.4 m.
    """
    assert "1.5" in arch_text, (
        "Default horizontal distance '1.5' (metres) not found in the "
        "documentation. The STM32 section should mention the default "
        "target horizontalDistance=1.5m."
    )
    assert "1.4" in arch_text, (
        "Default target altitude '1.4' (metres) not found in the "
        "documentation. The STM32 section should mention the default "
        "target altitude=1.4m."
    )


# ---------------------------------------------------------------------------
# Test – Parameter tables present (at least 2 STM32 tables)
# ---------------------------------------------------------------------------

def test_stm32_parameter_tables_present(arch_text):
    """
    Verify the document contains at least 14 Markdown table separator rows
    in total (12 existed before STM32; the STM32 section must add at least
    2 new tables — one for compile-time parameters and one for runtime
    parameters).
    """
    separator_pattern = re.compile(r"^\|[\s\-:]+\|[\s\-:]+\|", re.MULTILINE)
    separators = separator_pattern.findall(arch_text)
    assert len(separators) >= 14, (
        f"Only {len(separators)} Markdown table separator(s) found; "
        f"expected at least 14. The STM32 section should add at least 2 "
        f"new tables: one listing compile-time parameters (from "
        f"frontnet_config.h) and one listing the 21 runtime-tunable "
        f"parameters (frontnet group)."
    )


# ---------------------------------------------------------------------------
# Test – Minimum total line count after STM32 expansion
# ---------------------------------------------------------------------------

def test_stm32_minimum_total_lines(arch_text):
    """
    Verify that after adding the STM32 section the documentation file
    has at least 700 lines (significant expansion from 487 lines).
    """
    lines = arch_text.splitlines()
    assert len(lines) >= 700, (
        f"Architecture documentation is too short: expected >= 700 lines "
        f"after adding the STM32 section, but got {len(lines)}. "
        f"Expand the STM32 component documentation."
    )
