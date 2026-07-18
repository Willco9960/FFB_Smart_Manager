"""Standalone Tkinter shell for the Fantasy Football AI Manager.

This first UI intentionally uses placeholder content. It establishes the
navigation and visual structure without coupling the desktop shell to ESPN,
the simulator, or any trained policy checkpoints.
"""

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

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


class FantasyManagerApp(tk.Tk):
    """Main desktop window for the initial UI shell."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Fantasy Football AI Manager")
        self.geometry("1180x760")
        self.minsize(980, 640)
        self.configure(bg=COLORS["background"])
        self._configure_styles()
        self._build_shell()
        self.show_view("home")

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
        tk.Label(
            footer,
            text="No league connected yet",
            bg=COLORS["surface"],
            fg=COLORS["warning"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 0))

        self.content = ttk.Frame(shell, style="Content.TFrame")
        self.content.pack(side="left", fill="both", expand=True)

    def show_view(self, key: str) -> None:
        """Display a view and update the active navigation button."""

        if key not in VIEW_BY_KEY:
            raise ValueError(f"Unknown UI view: {key}")

        for view_key, button in self.nav_buttons.items():
            button.configure(
                style="SidebarActive.TButton" if view_key == key else "Sidebar.TButton"
            )
        for child in self.content.winfo_children():
            child.destroy()

        view = VIEW_BY_KEY[key]
        if key == "home":
            self._render_home(view)
        else:
            self._render_placeholder(view)

    def _render_header(self, view: ViewDefinition) -> None:
        header = tk.Frame(self.content, bg=COLORS["background"])
        header.pack(fill="x", padx=42, pady=(34, 24))
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

        hero = tk.Frame(body, bg=COLORS["accent_dark"])
        hero.pack(fill="x", pady=(0, 18))
        tk.Label(
            hero,
            text="Your assistant coach is ready to be connected.",
            bg=COLORS["accent_dark"],
            fg=COLORS["text"],
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=24, pady=(20, 4))
        tk.Label(
            hero,
            text=(
                "This shell is intentionally offline. Connect ESPN and trained "
                "models in a later milestone."
            ),
            bg=COLORS["accent_dark"],
            fg="#b8f7ef",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=24, pady=(0, 20))

        cards = tk.Frame(body, bg=COLORS["background"])
        cards.pack(fill="x", pady=(0, 22))
        cards.grid_columnconfigure((0, 1, 2), weight=1, uniform="cards")
        self._create_card(cards, 0, "ESPN CONNECTION", "Not connected", "Add read-only sync later")
        self._create_card(cards, 1, "CURRENT WEEK", "Preseason", "No league selected")
        self._create_card(cards, 2, "ASSISTANT STATUS", "UI prototype", "Model integration pending")

        section = tk.Frame(body, bg=COLORS["background"])
        section.pack(fill="both", expand=True)
        tk.Label(
            section,
            text="Explore the workspace",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 12))
        tk.Label(
            section,
            text="Use the navigation on the left to preview each major assistant area.",
            bg=COLORS["background"],
            fg=COLORS["muted"],
            font=("Segoe UI", 10),
        ).pack(anchor="w")

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
            text="This workspace is ready for the next implementation milestone.",
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
