"""Standalone Tkinter shell for the Fantasy Football AI Manager.

This first UI intentionally uses placeholder content. It establishes the
navigation and visual structure without coupling the desktop shell to ESPN,
the simulator, or any trained policy checkpoints.
"""

import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import ttk

from ui.demo_content import (
    DRAFT_RECOMMENDATIONS,
    LEAGUE_TRENDS,
    LINEUP_DECISIONS,
    MODEL_STATUSES,
    NEWS_UPDATES,
    PROJECTED_LEAGUE_POINTS,
    SEASON_WEEKS,
    SOCIAL_UPDATES,
    TRADE_IDEAS,
    WAIVER_TARGETS,
)
from ui.league_state import LeagueSummary, LeagueWorkspace

COLORS = {
    "background": "#0f172a",
    "surface": "#172033",
    "surface_alt": "#202d44",
    "border": "#2d3b55",
    "text": "#f8fafc",
    "muted": "#9aa9c2",
    "accent": "#5eead4",
    "accent_dark": "#123c4a",
    "warning": "#fbbf24",
}


@dataclass(frozen=True)
class ViewDefinition:
    key: str
    title: str
    subtitle: str
    description: str


VIEW_DEFINITIONS = (
    ViewDefinition(
        "home",
        "Home",
        "Your command center",
        "A quick view of team health, upcoming decisions, and assistant status.",
    ),
    ViewDefinition(
        "draft",
        "Draft Assistant",
        "Build your roster with confidence",
        "Live draft recommendations, roster needs, tiers, and positional scarcity.",
    ),
    ViewDefinition(
        "lineup",
        "Lineup Coach",
        "Make the best start/sit decisions",
        "Compare your current lineup with the strongest legal lineup for the week.",
    ),
    ViewDefinition(
        "waivers",
        "Waiver Wire",
        "Find the next useful addition",
        "Surface add/drop ideas, watchlist players, and short-term roster value.",
    ),
    ViewDefinition(
        "trades",
        "Trade Center",
        "Improve your roster without guesswork",
        "Review fair trade ideas and see how each side could benefit.",
    ),
    ViewDefinition(
        "models",
        "Models & Reports",
        "Understand what the assistant sees",
        "Inspect projections, confidence, training progress, and model accuracy.",
    ),
    ViewDefinition(
        "settings",
        "Settings",
        "Configure your assistant",
        "League connection, scoring rules, risk preferences, and display options.",
    ),
)

VIEW_BY_KEY = {view.key: view for view in VIEW_DEFINITIONS}
OPERATIONAL_VIEWS = {"draft", "lineup", "waivers", "trades"}


class FantasyManagerApp(tk.Tk):
    """Main desktop window for the initial UI shell."""

    def __init__(self) -> None:
        super().__init__()
        self.league_workspace = LeagueWorkspace()
        self.activity_var = tk.StringVar(value="Systems nominal · awaiting operator input")
        self.clock_var = tk.StringVar()
        self.current_view = "home"
        self.title("Fantasy Football AI Manager")
        self.geometry("1180x760")
        self.minsize(820, 640)
        self._layout_mode = "wide"
        self.configure(bg=COLORS["background"])
        self._configure_styles()
        self._build_shell()
        self.bind("<Configure>", self._on_resize)
        self.show_view("home")
        self._update_scope_status()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("App.TFrame", background=COLORS["background"])
        style.configure("Sidebar.TFrame", background=COLORS["surface"])
        style.configure("Content.TFrame", background=COLORS["background"])
        style.configure(
            "Sidebar.TButton",
            background=COLORS["surface"],
            foreground=COLORS["muted"],
            borderwidth=0,
            anchor="w",
            padding=(18, 11),
            font=("Segoe UI", 10),
        )
        style.map(
            "Sidebar.TButton",
            background=[("active", COLORS["surface_alt"])],
            foreground=[("active", COLORS["text"])],
        )
        style.configure(
            "SidebarActive.TButton",
            background=COLORS["accent_dark"],
            foreground=COLORS["accent"],
            borderwidth=0,
            anchor="w",
            padding=(18, 11),
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground="#082f35",
            borderwidth=0,
            padding=(14, 9),
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Demo.Treeview",
            background=COLORS["surface"],
            fieldbackground=COLORS["surface"],
            foreground=COLORS["text"],
            rowheight=34,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Demo.Treeview.Heading",
            background=COLORS["surface_alt"],
            foreground=COLORS["muted"],
            borderwidth=0,
            font=("Segoe UI", 8, "bold"),
        )
        style.map("Demo.Treeview", background=[("selected", COLORS["accent_dark"])])

    def _build_shell(self) -> None:
        shell = ttk.Frame(self, style="App.TFrame")
        shell.pack(fill="both", expand=True)

        self.sidebar = ttk.Frame(shell, width=245, style="Sidebar.TFrame")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg=COLORS["surface"])
        brand.pack(fill="x", padx=20, pady=(26, 28))
        tk.Label(
            brand,
            text="GRIDIRON",
            bg=COLORS["surface"],
            fg=COLORS["accent"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="AI MANAGER",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 19, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="Assistant coach workspace",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 0))

        self.nav_buttons: dict[str, ttk.Button] = {}
        for view in VIEW_DEFINITIONS:
            button = ttk.Button(
                self.sidebar,
                text=view.title,
                style="Sidebar.TButton",
                command=lambda key=view.key: self.show_view(key),
            )
            button.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[view.key] = button

        footer = tk.Frame(self.sidebar, bg=COLORS["surface"])
        footer.pack(side="bottom", fill="x", padx=20, pady=22)
        tk.Label(
            footer,
            text="LOCAL WORKSPACE",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")
        self.scope_status_label = tk.Label(
            footer,
            text="No league connected yet",
            bg=COLORS["surface"],
            fg=COLORS["warning"],
            font=("Segoe UI", 9),
        )
        self.scope_status_label.pack(anchor="w", pady=(4, 0))

        workspace = ttk.Frame(shell, style="Content.TFrame")
        workspace.pack(side="left", fill="both", expand=True)
        self._build_cockpit_bar(workspace)
        self.content = ttk.Frame(workspace, style="Content.TFrame")
        self.content.pack(fill="both", expand=True)
        self._update_clock()

    def _build_cockpit_bar(self, parent: ttk.Frame) -> None:
        bar = tk.Frame(parent, bg=COLORS["surface_alt"], height=54)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(
            bar,
            text="● LIVE DEMO",
            bg=COLORS["surface_alt"],
            fg=COLORS["accent"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(22, 12))
        tk.Label(
            bar,
            textvariable=self.clock_var,
            bg=COLORS["surface_alt"],
            fg=COLORS["muted"],
            font=("Consolas", 9),
        ).pack(side="left")
        tk.Label(
            bar,
            textvariable=self.activity_var,
            bg=COLORS["surface_alt"],
            fg=COLORS["text"],
            font=("Segoe UI", 9),
        ).pack(side="left", padx=28)
        ttk.Button(
            bar,
            text="Run Weekly Scan",
            style="Accent.TButton",
            command=self._run_weekly_scan,
        ).pack(side="right", padx=(8, 18))
        ttk.Button(
            bar,
            text="Sync All Demo Leagues",
            style="Accent.TButton",
            command=self._sync_demo_leagues,
        ).pack(side="right", padx=8)

    def _update_clock(self) -> None:
        self.clock_var.set(datetime.now().strftime("%a %b %d  ·  %I:%M:%S %p"))
        self.after(1000, self._update_clock)

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        next_mode = "compact" if event.width < 1050 else "wide"
        if next_mode == self._layout_mode:
            return
        self._layout_mode = next_mode
        self.sidebar.configure(width=205 if next_mode == "compact" else 245)
        if hasattr(self, "content"):
            self.show_view(self.current_view)

    def _set_activity(self, text: str) -> None:
        self.activity_var.set(text)

    def _sync_demo_leagues(self) -> None:
        self._set_activity("Syncing 4 demo league feeds...")
        self.after(
            650, lambda: self._set_activity("Sync complete · 4 leagues healthy · no live APIs used")
        )

    def _run_weekly_scan(self) -> None:
        self._set_activity("Scanning matchup, injury, waiver, and trade signals...")
        self.after(
            850, lambda: self._set_activity("Weekly scan complete · 6 recommendations queued")
        )

    def _acknowledge_action(self, action: str) -> None:
        self._set_activity(f"Demo action queued · {action}")

    def show_view(self, key: str) -> None:
        """Display a view and update the active navigation button."""

        if key not in VIEW_BY_KEY:
            raise ValueError(f"Unknown UI view: {key}")
        if key in OPERATIONAL_VIEWS and self.league_workspace.active_league is None:
            self._set_activity("Focus a league before opening operational controls")
            key = "home"
        self.current_view = key

        for view_key, button in self.nav_buttons.items():
            button.configure(
                style="SidebarActive.TButton" if view_key == key else "Sidebar.TButton"
            )
        for child in self.content.winfo_children():
            child.destroy()

        view = VIEW_BY_KEY[key]
        self._set_activity(f"Console routed to {view.title}")
        if key == "home":
            self._render_home(view)
        elif key == "draft":
            self._render_draft(view)
        elif key == "lineup":
            self._render_lineup(view)
        elif key == "waivers":
            self._render_waivers(view)
        elif key == "trades":
            self._render_trades(view)
        elif key == "models":
            self._render_models(view)
        elif key == "settings":
            self._render_settings(view)
        else:
            self._render_placeholder(view)

    def open_league(self, league_id: str) -> None:
        """Focus the UI on one league while keeping all others connected."""

        self.league_workspace.select(league_id)
        self._update_scope_status()
        self.show_view("home")

    def show_all_leagues(self) -> None:
        """Return to the cross-platform overview."""

        self.league_workspace.clear_selection()
        self._update_scope_status()
        self.show_view("home")

    def _update_scope_status(self) -> None:
        active_league = self.league_workspace.active_league
        if active_league is None:
            text = f"{len(self.league_workspace.leagues)} leagues connected"
        else:
            text = f"Focused: {active_league.league_name}"
        self.scope_status_label.configure(text=text)
        for view_key in OPERATIONAL_VIEWS:
            self.nav_buttons[view_key].configure(
                state="normal" if active_league is not None else "disabled"
            )

    def _render_header(self, view: ViewDefinition) -> None:
        header = tk.Frame(self.content, bg=COLORS["background"])
        header.pack(fill="x", padx=42, pady=(34, 24))
        scope = tk.Frame(header, bg=COLORS["background"])
        scope.pack(fill="x", pady=(0, 14))
        active_league = self.league_workspace.active_league
        if active_league is None:
            scope_text = "ALL CONNECTED LEAGUES"
            scope_color = COLORS["accent"]
        else:
            scope_text = f"FOCUSED LEAGUE · {active_league.display_name.upper()}"
            scope_color = COLORS["warning"]
        tk.Label(
            scope,
            text=scope_text,
            bg=COLORS["background"],
            fg=scope_color,
            font=("Segoe UI", 8, "bold"),
        ).pack(side="left")
        if active_league is not None:
            ttk.Button(
                scope,
                text="View All Leagues",
                style="Accent.TButton",
                command=self.show_all_leagues,
            ).pack(side="right")
        tk.Label(
            header,
            text=view.title,
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 26, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text=view.subtitle,
            bg=COLORS["background"],
            fg=COLORS["accent"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(5, 0))
        tk.Label(
            header,
            text=view.description,
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 0))

    def _render_home(self, view: ViewDefinition) -> None:
        self._render_header(view)
        body = tk.Frame(self.content, bg=COLORS["background"])
        body.pack(fill="both", expand=True, padx=42, pady=(0, 34))

        if self.league_workspace.active_league is not None:
            self._render_focused_league_home(body, self.league_workspace.active_league)
            return

        hero = tk.Frame(body, bg=COLORS["accent_dark"])
        hero.pack(fill="x", pady=(0, 18))
        tk.Label(
            hero,
            text="Your multi-league assistant workspace is online.",
            bg=COLORS["accent_dark"],
            fg=COLORS["text"],
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=24, pady=(20, 4))
        tk.Label(
            hero,
            text=(
                "Demo connections are active across ESPN, Sleeper, and NFL Fantasy. "
                "Live sync will replace this data later."
            ),
            bg=COLORS["accent_dark"],
            fg="#b8f7ef",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=24, pady=(0, 20))

        cards = tk.Frame(body, bg=COLORS["background"])
        cards.pack(fill="x", pady=(0, 22))
        cards.grid_columnconfigure((0, 1, 2), weight=1, uniform="cards")
        self._create_card(
            cards, 0, "CONNECTED LEAGUES", "4 leagues", "ESPN · Sleeper · NFL Fantasy"
        )
        self._create_card(cards, 1, "CURRENT WEEK", "Week 12", "Demo season snapshot")
        self._create_card(
            cards, 2, "ASSISTANT STATUS", "Online (demo)", "Models and reports simulated"
        )

        section = tk.Frame(body, bg=COLORS["background"])
        section.pack(fill="both", expand=True)
        tk.Label(
            section,
            text="Connected leagues",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 12))
        tk.Label(
            section,
            text="Select a league to focus every assistant view on that platform and roster.",
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(0, 12))
        self._render_league_rows(section)
        tk.Label(
            section,
            text="Season intelligence",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(18, 10))
        chart_row = tk.Frame(section, bg=COLORS["background"])
        chart_row.pack(fill="x")
        chart_row.grid_columnconfigure((0, 1), weight=1, uniform="charts")
        self._create_line_chart(chart_row, "Points-for trajectory", LEAGUE_TRENDS, column=0)
        self._create_projection_chart(
            chart_row,
            "Projected points this week",
            PROJECTED_LEAGUE_POINTS,
            column=1,
        )

    def _render_league_rows(self, parent: tk.Frame) -> None:
        panel = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        panel.pack(fill="x")
        for row, league in enumerate(self.league_workspace.leagues):
            panel.grid_rowconfigure(row, weight=1)
            self._create_league_row(panel, row, league)

    def _create_league_row(self, parent: tk.Frame, row: int, league: LeagueSummary) -> None:
        row_frame = tk.Frame(parent, bg=COLORS["surface"])
        row_frame.grid(row=row, column=0, sticky="ew", padx=18, pady=(14 if row == 0 else 7, 7))
        row_frame.grid_columnconfigure(1, weight=1)
        tk.Label(
            row_frame,
            text=league.platform.upper(),
            bg=COLORS["surface"],
            fg=COLORS["accent"],
            font=("Segoe UI", 8, "bold"),
            width=12,
            anchor="w",
        ).grid(row=0, column=0, rowspan=2, sticky="nw")
        tk.Label(
            row_frame,
            text=league.league_name,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).grid(row=0, column=1, sticky="w")
        tk.Label(
            row_frame,
            text=f"{league.team_name} · {league.record} · {league.matchup}",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
            anchor="w",
        ).grid(row=1, column=1, sticky="w")
        tk.Label(
            row_frame,
            text=f"Proj. {league.projected_points:.1f}",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=2, rowspan=2, padx=18)
        ttk.Button(
            row_frame,
            text="Open League",
            style="Accent.TButton",
            command=lambda league_id=league.league_id: self.open_league(league_id),
        ).grid(row=0, column=3, rowspan=2)

    def _render_focused_league_home(self, body: tk.Frame, league: LeagueSummary) -> None:
        hero = tk.Frame(body, bg=COLORS["accent_dark"])
        hero.pack(fill="x", pady=(0, 18))
        tk.Label(
            hero,
            text=f"{league.team_name} · {league.league_name}",
            bg=COLORS["accent_dark"],
            fg=COLORS["text"],
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=24, pady=(20, 4))
        tk.Label(
            hero,
            text=f"{league.platform} · {league.matchup} · {league.sync_status}",
            bg=COLORS["accent_dark"],
            fg="#b8f7ef",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=24, pady=(0, 20))

        cards = tk.Frame(body, bg=COLORS["background"])
        cards.pack(fill="x", pady=(0, 22))
        cards.grid_columnconfigure((0, 1, 2), weight=1, uniform="cards")
        self._create_card(cards, 0, "RECORD", league.record, "Current season")
        self._create_card(cards, 1, "POINTS FOR", f"{league.points_for:.1f}", "Scored so far")
        self._create_card(
            cards, 2, "MATCHUP PROJECTION", f"{league.projected_points:.1f}", league.matchup
        )
        trend = next(
            (series for series in LEAGUE_TRENDS if series[0] == league.league_name),
            LEAGUE_TRENDS[0],
        )
        self._create_line_chart(body, "Focused league trajectory", (trend,))

        ttk.Button(
            body,
            text="Back to All Leagues",
            style="Accent.TButton",
            command=self.show_all_leagues,
        ).pack(anchor="w")

    def _create_line_chart(
        self,
        parent: tk.Frame,
        title: str,
        series: tuple[tuple[str, tuple[int, ...], str], ...],
        column: int | None = None,
    ) -> None:
        panel = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        if column is None:
            panel.pack(fill="x", pady=(0, 18))
        else:
            panel.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 6))
        tk.Label(
            panel,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        canvas = tk.Canvas(panel, height=190, bg=COLORS["surface"], highlightthickness=0)
        canvas.pack(fill="x", padx=10, pady=(0, 10))

        def draw(_event: tk.Event | None = None) -> None:
            canvas.delete("all")
            width = max(canvas.winfo_width(), 250)
            height = max(canvas.winfo_height(), 180)
            left, right, top, bottom = 36, 12, 16, 28
            plot_width = width - left - right
            plot_height = height - top - bottom
            all_values = [value for _label, values, _color in series for value in values]
            minimum = min(all_values) - 5
            maximum = max(all_values) + 5
            value_range = max(maximum - minimum, 1)
            for grid_line in range(4):
                y = top + plot_height * grid_line / 3
                canvas.create_line(left, y, width - right, y, fill=COLORS["border"])
                label_value = maximum - value_range * grid_line / 3
                canvas.create_text(
                    4,
                    y,
                    text=f"{label_value:.0f}",
                    anchor="w",
                    fill=COLORS["muted"],
                    font=("Segoe UI", 7),
                )
            for index, week in enumerate(SEASON_WEEKS):
                x = left + plot_width * index / max(len(SEASON_WEEKS) - 1, 1)
                canvas.create_text(
                    x,
                    height - 10,
                    text=week,
                    fill=COLORS["muted"],
                    font=("Segoe UI", 7),
                )
            legend_x = left
            for label, values, color in series:
                points = []
                for index, value in enumerate(values):
                    x = left + plot_width * index / max(len(values) - 1, 1)
                    y = top + (maximum - value) / value_range * plot_height
                    points.extend((x, y))
                canvas.create_line(*points, fill=color, width=2, smooth=True)
                for index in range(0, len(points), 2):
                    canvas.create_oval(
                        points[index] - 3,
                        points[index + 1] - 3,
                        points[index] + 3,
                        points[index + 1] + 3,
                        fill=color,
                        outline=color,
                    )
                canvas.create_oval(legend_x, 3, legend_x + 7, 10, fill=color, outline=color)
                canvas.create_text(
                    legend_x + 11,
                    6,
                    text=label,
                    anchor="w",
                    fill=COLORS["muted"],
                    font=("Segoe UI", 7),
                )
                legend_x += 90 + len(label) * 3

        canvas.bind("<Configure>", draw)
        parent.after_idle(draw)

    def _create_projection_chart(
        self,
        parent: tk.Frame,
        title: str,
        values: tuple[tuple[str, float], ...],
        column: int,
    ) -> None:
        panel = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        panel.grid(row=0, column=column, sticky="nsew", padx=(6, 0))
        tk.Label(
            panel,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        canvas = tk.Canvas(panel, height=190, bg=COLORS["surface"], highlightthickness=0)
        canvas.pack(fill="x", padx=10, pady=(0, 10))

        def draw(_event: tk.Event | None = None) -> None:
            canvas.delete("all")
            width = max(canvas.winfo_width(), 250)
            max_value = max(value for _label, value in values)
            baseline = 160
            spacing = (width - 24) / len(values)
            bar_width = max(spacing - 12, 24)
            for index, (label, value) in enumerate(values):
                x = 18 + index * spacing
                bar_height = 115 * value / max_value
                canvas.create_rectangle(
                    x,
                    baseline - bar_height,
                    x + bar_width,
                    baseline,
                    fill=COLORS["accent"],
                    outline="",
                )
                canvas.create_text(
                    x + bar_width / 2,
                    baseline + 12,
                    text=label,
                    fill=COLORS["muted"],
                    font=("Segoe UI", 7),
                )
                canvas.create_text(
                    x + bar_width / 2,
                    baseline - bar_height - 9,
                    text=f"{value:.1f}",
                    fill=COLORS["text"],
                    font=("Segoe UI", 8, "bold"),
                )
            canvas.create_line(12, baseline, width - 12, baseline, fill=COLORS["border"])

        canvas.bind("<Configure>", draw)
        parent.after_idle(draw)

    def _create_card(
        self, parent: tk.Frame, column: int, label: str, value: str, detail: str
    ) -> None:
        card = tk.Frame(
            parent, bg=COLORS["surface"], highlightbackground=COLORS["border"], highlightthickness=1
        )
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 7, 7))
        tk.Label(
            card, text=label, bg=COLORS["surface"], fg=COLORS["muted"], font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=18, pady=(17, 5))
        tk.Label(
            card, text=value, bg=COLORS["surface"], fg=COLORS["text"], font=("Segoe UI", 15, "bold")
        ).pack(anchor="w", padx=18)
        tk.Label(
            card, text=detail, bg=COLORS["surface"], fg=COLORS["muted"], font=("Segoe UI", 9)
        ).pack(anchor="w", padx=18, pady=(5, 17))

    def _create_demo_banner(self, parent: tk.Frame, text: str) -> None:
        banner = tk.Frame(parent, bg="#302a17")
        banner.pack(fill="x", pady=(0, 16))
        tk.Label(
            banner,
            text=f"DEMO PREVIEW  ·  {text}",
            bg="#302a17",
            fg=COLORS["warning"],
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=16, pady=10)

    def _create_kpi_cards(self, parent: tk.Frame, cards: list[tuple[str, str, str]]) -> None:
        frame = tk.Frame(parent, bg=COLORS["background"])
        frame.pack(fill="x", pady=(0, 18))
        frame.grid_columnconfigure(tuple(range(len(cards))), weight=1, uniform="kpis")
        for index, (label, value, detail) in enumerate(cards):
            self._create_card(frame, index, label, value, detail)

    def _create_table(
        self,
        parent: tk.Frame,
        columns: tuple[str, ...],
        rows: list[tuple[str, ...]],
        widths: tuple[int, ...],
        action_label: str = "Queue Demo Action",
    ) -> ttk.Treeview:
        table = ttk.Treeview(
            parent,
            columns=columns,
            show="headings",
            style="Demo.Treeview",
            height=min(max(len(rows), 4), 8),
        )
        for column, width in zip(columns, widths, strict=True):
            table.heading(column, text=column.upper())
            table.column(column, width=width, anchor="w", stretch=True)
        for row in rows:
            table.insert("", "end", values=row)
        table.pack(fill="x", padx=1, pady=1)

        detail_var = tk.StringVar(value="Select a row to inspect its recommendation details.")
        detail_panel = tk.Frame(parent, bg=COLORS["surface_alt"])
        detail_panel.pack(fill="x", pady=(8, 0))
        tk.Label(
            detail_panel,
            textvariable=detail_var,
            bg=COLORS["surface_alt"],
            fg=COLORS["muted"],
            anchor="w",
            font=("Segoe UI", 9),
        ).pack(side="left", fill="x", expand=True, padx=14, pady=10)

        def queue_selected_action() -> None:
            selected = table.selection()
            if not selected:
                self._acknowledge_action("select a row first")
                return
            values = table.item(selected[0], "values")
            self._acknowledge_action(str(values[1] if len(values) > 1 else values[0]))

        ttk.Button(
            detail_panel,
            text=action_label,
            style="Accent.TButton",
            command=queue_selected_action,
        ).pack(side="right", padx=10, pady=6)

        def update_details(_event: tk.Event) -> None:
            selected = table.selection()
            if selected:
                values = table.item(selected[0], "values")
                subject = values[1] if len(values) > 1 else values[0]
                detail_var.set(f"Selected: {subject}  ·  {values[-1]}")

        table.bind("<<TreeviewSelect>>", update_details)
        return table

    def _render_draft(self, view: ViewDefinition) -> None:
        self._render_header(view)
        body = tk.Frame(self.content, bg=COLORS["background"])
        body.pack(fill="both", expand=True, padx=42, pady=(0, 34))
        self._create_demo_banner(body, "Draft board is simulated; no platform draft is connected.")
        self._create_kpi_cards(
            body,
            [
                ("ON THE CLOCK", "Pick 7", "Round 4 · Snake draft"),
                ("ROSTER NEED", "WR", "Two starting spots remain"),
                ("TOP RECOMMENDATION", "CeeDee Lamb", "18.7 projected points"),
            ],
        )
        tk.Label(
            body,
            text="Recommended picks",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        rows = [
            (
                str(item.rank),
                item.player,
                item.position,
                f"{item.projection:.1f}",
                item.value,
                item.reason,
            )
            for item in DRAFT_RECOMMENDATIONS
        ]
        self._create_table(
            body,
            ("Rank", "Player", "Pos", "Projection", "Value", "Why"),
            rows,
            (55, 150, 55, 95, 105, 360),
            action_label="Queue draft pick",
        )

    def _create_lineup_overlay(self, parent: tk.Frame) -> None:
        panel = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(
            panel,
            text="Lineup projection overlay",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 3))
        tk.Label(
            panel,
            text="Gray = current lineup   ·   Teal = suggested lineup",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=14, pady=(0, 5))
        canvas = tk.Canvas(panel, height=242, bg=COLORS["surface"], highlightthickness=0)
        canvas.pack(fill="x", padx=10, pady=(0, 10))

        def draw(_event: tk.Event | None = None) -> None:
            canvas.delete("all")
            width = max(canvas.winfo_width(), 300)
            max_projection = max(item.ai_projection for item in LINEUP_DECISIONS)
            left = 54
            track_width = max(width - left - 82, 150)
            for index, item in enumerate(LINEUP_DECISIONS):
                y = 22 + index * 29
                current_width = track_width * item.espn_projection / max_projection
                suggested_width = track_width * item.ai_projection / max_projection
                canvas.create_text(
                    4,
                    y,
                    text=item.slot,
                    anchor="w",
                    fill=COLORS["muted"],
                    font=("Segoe UI", 8, "bold"),
                )
                canvas.create_rectangle(
                    left, y - 7, left + track_width, y + 7, fill=COLORS["surface_alt"], outline=""
                )
                canvas.create_rectangle(
                    left, y - 7, left + current_width, y + 7, fill="#64748b", outline=""
                )
                canvas.create_line(
                    left, y + 10, left + suggested_width, y + 10, fill=COLORS["accent"], width=4
                )
                canvas.create_text(
                    width - 76,
                    y,
                    text=f"{item.espn_projection:.1f} → {item.ai_projection:.1f}",
                    anchor="w",
                    fill=COLORS["text"],
                    font=("Segoe UI", 8),
                )
            canvas.create_text(
                left, 232, text="Current", anchor="w", fill="#94a3b8", font=("Segoe UI", 8)
            )
            canvas.create_line(left + 48, 232, left + 76, 232, fill=COLORS["accent"], width=4)
            canvas.create_text(
                left + 82,
                232,
                text="Suggested",
                anchor="w",
                fill=COLORS["accent"],
                font=("Segoe UI", 8),
            )

        canvas.bind("<Configure>", draw)
        parent.after_idle(draw)

    def _create_feed_panel(self, parent: tk.Frame, title: str, items: tuple) -> None:
        panel = tk.Frame(
            parent,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        panel.pack(fill="x", pady=(0, 8))
        tk.Label(
            panel,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 5))
        for item in items:
            row = tk.Frame(panel, bg=COLORS["surface"])
            row.pack(fill="x", padx=14, pady=(0, 8))
            tk.Label(
                row,
                text=f"{item.source} · {item.timestamp}",
                bg=COLORS["surface"],
                fg=COLORS["accent"],
                font=("Segoe UI", 7, "bold"),
            ).pack(anchor="w")
            tk.Label(
                row,
                text=item.headline,
                bg=COLORS["surface"],
                fg=COLORS["text"],
                font=("Segoe UI", 9, "bold"),
                wraplength=280,
                justify="left",
            ).pack(anchor="w")
            tk.Label(
                row,
                text=item.detail,
                bg=COLORS["surface"],
                fg=COLORS["muted"],
                font=("Segoe UI", 8),
                wraplength=280,
                justify="left",
            ).pack(anchor="w")

    def _render_lineup(self, view: ViewDefinition) -> None:
        self._render_header(view)
        body = tk.Frame(self.content, bg=COLORS["background"])
        body.pack(fill="both", expand=True, padx=42, pady=(0, 34))
        self._create_demo_banner(
            body, "Lineup recommendations use simulated projections and matchup context."
        )
        self._create_kpi_cards(
            body,
            [
                ("CURRENT LINEUP", "112.8", "ESPN-style projection"),
                ("AI RECOMMENDATION", "119.6", "+6.8 projected points"),
                ("MATCHUP", "Favorable", "Projected win probability 62%"),
            ],
        )
        cockpit_row = tk.Frame(body, bg=COLORS["background"])
        cockpit_row.pack(fill="x", pady=(0, 16))
        cockpit_row.grid_columnconfigure(0, weight=3)
        cockpit_row.grid_columnconfigure(1, weight=2)
        self._create_lineup_overlay(cockpit_row)
        signal_column = tk.Frame(cockpit_row, bg=COLORS["background"])
        signal_column.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._create_feed_panel(signal_column, "News & Injury Radar", NEWS_UPDATES)
        self._create_feed_panel(signal_column, "Social Pulse", SOCIAL_UPDATES)
        tk.Label(
            body,
            text="Start / sit board",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        rows = [
            (
                item.slot,
                item.player,
                item.opponent,
                f"{item.espn_projection:.1f}",
                f"{item.ai_projection:.1f}",
                item.decision,
                item.reason,
            )
            for item in LINEUP_DECISIONS
        ]
        self._create_table(
            body,
            ("Slot", "Player", "Opp", "ESPN", "AI", "Decision", "Reason"),
            rows,
            (55, 130, 75, 65, 65, 95, 330),
            action_label="Queue lineup change",
        )

    def _render_waivers(self, view: ViewDefinition) -> None:
        self._render_header(view)
        body = tk.Frame(self.content, bg=COLORS["background"])
        body.pack(fill="both", expand=True, padx=42, pady=(0, 34))
        self._create_demo_banner(
            body, "Waiver alerts are simulated; add/drop actions are not submitted."
        )
        self._create_kpi_cards(
            body,
            [
                ("AVAILABLE PLAYERS", "148", "Demo free-agent pool"),
                ("ADD NOW", "2", "High-priority upgrades"),
                ("WATCHLIST", "11", "Monitor for role changes"),
            ],
        )
        tk.Label(
            body,
            text="Waiver recommendations",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        rows = [
            (
                item.action,
                item.player,
                item.position,
                f"{item.projection:.1f}",
                item.confidence,
                item.reason,
            )
            for item in WAIVER_TARGETS
        ]
        self._create_table(
            body,
            ("Action", "Player", "Pos", "3-Wk Proj.", "Confidence", "Reason"),
            rows,
            (120, 140, 55, 90, 95, 360),
            action_label="Queue waiver move",
        )

    def _render_trades(self, view: ViewDefinition) -> None:
        self._render_header(view)
        body = tk.Frame(self.content, bg=COLORS["background"])
        body.pack(fill="both", expand=True, padx=42, pady=(0, 34))
        self._create_demo_banner(
            body,
            "Trade ideas are simulated and require your approval before any future submission.",
        )
        self._create_kpi_cards(
            body,
            [
                ("TEAM NEED", "WR depth", "Highest projected gap"),
                ("SURPLUS", "RB / TE", "Potential trade capital"),
                ("BEST FIT", "+4.6", "Projected weekly gain"),
            ],
        )
        tk.Label(
            body,
            text="Potential trade ideas",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        rows = [
            (
                item.proposal,
                item.send,
                item.receive,
                item.target_team,
                f"+{item.weekly_delta:.1f}",
                f"{item.fairness}/100",
            )
            for item in TRADE_IDEAS
        ]
        self._create_table(
            body,
            ("Type", "Send", "Receive", "Target Team", "Your Delta", "Fairness"),
            rows,
            (90, 150, 170, 170, 90, 90),
            action_label="Queue trade review",
        )

    def _render_models(self, view: ViewDefinition) -> None:
        self._render_header(view)
        body = tk.Frame(self.content, bg=COLORS["background"])
        body.pack(fill="both", expand=True, padx=42, pady=(0, 34))
        self._create_demo_banner(
            body, "Model metrics are illustrative until local checkpoints are connected."
        )
        self._create_kpi_cards(
            body,
            [
                ("MODEL STATUS", "4 loaded", "Demo checkpoints"),
                ("PROJECTION MAE", "18.7", "Illustrative points"),
                ("TRAINING DATA", "2001–2024", "Historical seasons"),
            ],
        )
        tk.Label(
            body,
            text="Connected services and model health",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        rows = [(item.model, item.status, item.last_run, item.detail) for item in MODEL_STATUSES]
        self._create_table(
            body,
            ("Model / Service", "Status", "Last Run", "Details"),
            rows,
            (200, 120, 140, 380),
            action_label="Inspect model",
        )

    def _render_settings(self, view: ViewDefinition) -> None:
        self._render_header(view)
        body = tk.Frame(self.content, bg=COLORS["background"])
        body.pack(fill="both", expand=True, padx=42, pady=(0, 34))
        self._create_demo_banner(
            body, "These controls preview the future connection and league configuration flow."
        )
        tk.Label(
            body,
            text="Platform connections",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        connection_panel = tk.Frame(body, bg=COLORS["surface"])
        connection_panel.pack(fill="x", pady=(0, 22))
        for row, platform in enumerate(("ESPN", "Sleeper", "NFL Fantasy")):
            tk.Label(
                connection_panel,
                text=platform,
                bg=COLORS["surface"],
                fg=COLORS["text"],
                font=("Segoe UI", 10, "bold"),
                width=18,
                anchor="w",
            ).grid(row=row, column=0, padx=18, pady=12, sticky="w")
            tk.Label(
                connection_panel,
                text="Demo connected",
                bg=COLORS["surface"],
                fg=COLORS["warning"],
                font=("Segoe UI", 9),
            ).grid(row=row, column=1, padx=18, sticky="w")
            ttk.Button(
                connection_panel,
                text="Replace later",
                style="Accent.TButton",
            ).grid(row=row, column=2, padx=18, pady=8)

        tk.Label(
            body,
            text="League defaults",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 10))
        rows = [
            ("Roster size", "16 players", "ESPN-style default"),
            (
                "Starting lineup",
                "QB · 2 RB · 2 WR · TE · FLEX · K · DST",
                "League-specific override later",
            ),
            ("Assistant mode", "Recommend + explain", "Human approval required"),
            ("Risk preference", "Balanced", "Adjustable per league"),
        ]
        self._create_table(
            body,
            ("Setting", "Current", "Notes"),
            rows,
            (180, 350, 330),
            action_label="Edit demo setting",
        )

    def _render_placeholder(self, view: ViewDefinition) -> None:
        self._render_header(view)
        panel = tk.Frame(
            self.content,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        panel.pack(fill="both", expand=True, padx=42, pady=(0, 34))
        tk.Label(
            panel,
            text=f"{view.title} view",
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", padx=28, pady=(30, 8))
        tk.Label(
            panel,
            text=(
                "This workspace is ready for the next implementation milestone. "
                "Select a league from Home to focus its data here."
            ),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=("Segoe UI", 11),
        ).pack(anchor="w", padx=28)
        ttk.Button(
            panel,
            text="Return to Home",
            style="Accent.TButton",
            command=lambda: self.show_view("home"),
        ).pack(anchor="w", padx=28, pady=(24, 0))


def main() -> None:
    FantasyManagerApp().mainloop()


if __name__ == "__main__":
    main()
