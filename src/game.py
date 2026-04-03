"""
game.py — CLI game loop using the `rich` library.

Flow per case:
  1. Display case title + narrative
  2. Ask player for hint (optional)  →  entity-bridged hop-1 retrieval
  3. Ask player for a second hint    →  entity-bridged hop-2 retrieval
  4. Collect structured guess: who / how / where
  5. Verify each claim individually → display per-field SUPPORTED/REFUTED
  6. Reveal answer + briefing
"""

from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from src.data_loader import load_cases
from src.retriever import retrieve
from src.verifier import verify_claim

console = Console()

DIFFICULTY_COLOR = {
    "쉬움": "green",
    "쉬움-중간": "yellow",
    "중간": "yellow",
    "중간-어려움": "orange3",
    "어려움": "red",
}

FIELD_LABELS = {"who": "범인", "how": "수단", "where": "장소"}

LABEL_DISPLAY = {
    "SUPPORTED": ("[bold green]네 (O) — 사실[/bold green]", "green"),
    "REFUTED": ("[bold red]아니오 (X) — 거짓[/bold red]", "red"),
    "NOT_ENOUGH_INFO": ("[bold yellow]정보 부족[/bold yellow]", "yellow"),
}


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_title_screen() -> None:
    console.print()
    console.print(
        Panel(
            Text("살인 미스터리\n추리 게임", justify="center", style="bold white"),
            subtitle="[dim]FBI 실제 사건 기반 · Fact Verification · Powered by Ollama[/dim]",
            border_style="bright_red",
            padding=(1, 4),
        )
    )
    console.print()


def _print_case_header(case: dict[str, Any]) -> None:
    diff = case.get("difficulty", "중간")
    color = DIFFICULTY_COLOR.get(diff, "white")
    console.print(Rule(f"[bold]{case['title']}[/bold]", style="bright_red"))
    console.print(
        f"  [dim]{case.get('subtitle', '')}[/dim]   난이도: [{color}]{diff}[/{color}]"
    )
    console.print()


def _print_narrative(case: dict[str, Any]) -> None:
    console.print(
        Panel(
            case["narrative"],
            title="[bold yellow]사건 개요[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()


def _print_hop_clues(chain: list[dict[str, Any]], hop_label: str) -> None:
    target_hop = 1 if hop_label == "1차" else 2
    items = [c for c in chain if c.get("hop") == target_hop]
    if not items:
        console.print(f"[dim]{hop_label} 단서를 찾지 못했습니다.[/dim]")
        return
    for item in items:
        clue = item.get("clue", item)
        bridge = item.get("bridge_entity")
        bridge_str = (
            f" | 연결어: [bold magenta]{bridge}[/bold magenta]" if bridge else ""
        )
        console.print(
            Panel(
                clue.get("text", ""),
                title=f"[bold cyan]{hop_label} 단서[/bold cyan]  "
                f"[dim](hop {item.get('hop', '?')}){bridge_str}[/dim]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
    console.print()


def _print_verification_result(verdict: dict[str, Any]) -> None:
    """항목별 Fact Verification 결과를 표시."""
    console.print(Rule("[bold]진위 판정[/bold]", style="bright_cyan"))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("항목", width=8, justify="center")
    table.add_column("판정", width=18, justify="center")
    table.add_column("근거", min_width=30)

    fields = verdict.get("fields", {})
    for field in ["who", "how", "where"]:
        label_ko = FIELD_LABELS[field]
        fv = fields.get(field, {})
        label = fv.get("label", "NOT_ENOUGH_INFO")
        display_text, color = LABEL_DISPLAY.get(label, LABEL_DISPLAY["NOT_ENOUGH_INFO"])
        reasoning = fv.get("reasoning", "")
        table.add_row(label_ko, display_text, reasoning)

    console.print(table)

    # Score
    correct = verdict.get("correct_count", 0)
    total = verdict.get("total_count", 3)
    score = verdict.get("score", 0)
    score_color = "green" if score >= 66 else ("yellow" if score >= 33 else "red")
    console.print()
    console.print(
        f"  [bold]점수:[/bold] [{score_color}]{score}점[/{score_color}]  "
        f"({correct}/{total} 항목 일치)"
    )
    console.print()


def _print_answer_reveal(case: dict[str, Any]) -> None:
    answer = case.get("answer", {})
    console.print(Rule("[bold]사건의 진실[/bold]", style="bright_yellow"))

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("항목", style="bold", width=12)
    table.add_column("내용")

    table.add_row("범인", answer.get("who", "미확인"))
    if answer.get("mastermind"):
        table.add_row("배후", answer["mastermind"])
    table.add_row("수단", answer.get("how", "미확인"))
    table.add_row("장소", answer.get("where", "미확인"))
    table.add_row("동기", answer.get("motive", "미확인"))

    console.print(table)
    console.print()
    console.print(
        Panel(
            case.get("briefing", "브리핑 정보 없음."),
            title="[bold yellow]사건 브리핑[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()


def _collect_guess() -> dict[str, str]:
    console.print("[bold]추리를 입력하세요.[/bold] (잘 모르면 '모름'이라고 입력)")
    console.print()
    who = Prompt.ask("  [bold cyan]범인은 누구입니까?[/bold cyan]")
    how = Prompt.ask("  [bold cyan]어떤 수단/방법을 사용했습니까?[/bold cyan]")
    where = Prompt.ask("  [bold cyan]사건이 벌어진 장소는 어디입니까?[/bold cyan]")
    console.print()
    return {"who": who, "how": how, "where": where}


# ---------------------------------------------------------------------------
# Core game loop
# ---------------------------------------------------------------------------


def play_case(case: dict[str, Any]) -> dict[str, Any]:
    """Play through a single case. Returns the verdict dict."""
    _print_case_header(case)
    _print_narrative(case)

    # Hint rounds — entity-bridged retrieval
    for hint_round in range(1, 3):
        want_hint = Prompt.ask(
            f"  단서를 요청하시겠습니까? (힌트 {hint_round}/2)",
            choices=["예", "아니오"],
            default="예",
        )
        console.print()
        if want_hint == "아니오":
            break

        hint_query = Prompt.ask("  무엇에 대해 조사하시겠습니까? (자유롭게 입력)")
        console.print()

        results = retrieve(
            case=case,
            user_query=hint_query,
            hop1_k=2,
            hop2_k=2,
        )

        if hint_round == 1:
            _print_hop_clues(results["chain"], "1차")
        else:
            _print_hop_clues(results["chain"], "2차")

        if results["bridge_entities"]:
            console.print(
                f"  [dim]연결된 엔티티: {', '.join(results['bridge_entities'])}[/dim]\n"
            )

    # Collect guess
    guess = _collect_guess()

    # Evidence retrieval for verification
    console.print("[dim]증거 분석 중...[/dim]")
    evidence = retrieve(
        case=case,
        user_query=" ".join(guess.values()),
        hop1_k=3,
        hop2_k=2,
    )

    # Fact Verification — per-field
    console.print("[dim]진위 검증 중...[/dim]\n")
    verdict = verify_claim(case, guess, evidence["chain"])

    _print_verification_result(verdict)
    _print_answer_reveal(case)

    return verdict


def run_game() -> None:
    """Main entry point for the game."""
    _print_title_screen()

    cases = load_cases()

    console.print("[bold]사건 목록[/bold]")
    for i, case in enumerate(cases, 1):
        diff = case.get("difficulty", "중간")
        color = DIFFICULTY_COLOR.get(diff, "white")
        console.print(
            f"  [{color}]{i}.[/{color}] {case['title']}  "
            f"[dim]{case.get('subtitle', '')}[/dim]  "
            f"[{color}][{diff}][/{color}]"
        )
    console.print()

    choice = Prompt.ask(
        "  플레이할 사건 번호를 선택하세요 (전체 플레이: 0)",
        default="1",
    )
    console.print()

    if choice == "0":
        selected = cases
    else:
        try:
            idx = int(choice) - 1
            selected = [cases[idx]]
        except (ValueError, IndexError):
            console.print("[red]올바른 번호를 입력하세요.[/red]")
            return

    results_log: list[dict] = []
    for case in selected:
        verdict = play_case(case)
        results_log.append(verdict)
        if len(selected) > 1:
            score = verdict.get("score", 0)
            console.print(f"  [dim]이 사건 점수: {score}점[/dim]")
            cont = Prompt.ask(
                "  다음 사건으로 넘어가시겠습니까?",
                choices=["예", "아니오"],
                default="예",
            )
            console.print()
            if cont == "아니오":
                break

    if len(selected) > 1:
        avg = sum(v.get("score", 0) for v in results_log) // len(results_log)
        console.print(Rule("[bold]최종 결과[/bold]", style="bright_yellow"))
        console.print(f"  평균 점수: [bold yellow]{avg}점[/bold yellow] / 100점")

    console.print("\n[bold]수사를 마칩니다. 감사합니다.[/bold]\n")
