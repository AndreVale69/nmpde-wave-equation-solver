"""
A simple Tkinter GUI for creating and editing deal.II-style `.prm` files without any dependencies.

Run:
  python3 tools/prm_gui.py

Features:
- Edit sections: Problem, Boundary condition, Mesh, Time, Output
- Load sensible defaults
- Load a `.prm` file (simple parser)
- Save `.prm` file in the project's expected `set key = value` format
- Preview the resulting file before saving
"""
import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "PRM Editor"

THEMES = {
    "light": {
        "bg": "#f6f7fb",
        "panel": "#ffffff",
        "text": "#1f2430",
        "muted": "#5f6b7a",
        "accent": "#3b82f6",
        "accent_dark": "#2563eb",
        "border": "#d7dbe6",
        "entry": "#ffffff",
        "preview_bg": "#0f172a",
        "preview_fg": "#e2e8f0",
        "select_bg": "#dbeafe",
        "select_fg": "#0f172a",
        "listbox_bg": "#ffffff",
        "listbox_fg": "#1f2430",
        "listbox_select_bg": "#bfdbfe",
        "listbox_select_fg": "#0f172a",
        "disabled_entry": "#eef0f5",
        "focus_border": "#9aa3b2",
        "enabled_border": "#9aa3b2",
        "disabled_border": "#d7dbe6",
    },
    "dark": {
        "bg": "#0b1220",
        "panel": "#0f172a",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "accent": "#5b9dff",
        "accent_dark": "#3b82f6",
        "border": "#1e293b",
        "entry": "#0f172a",
        "preview_bg": "#0b1220",
        "preview_fg": "#e2e8f0",
        "select_bg": "#334155",
        "select_fg": "#e2e8f0",
        "listbox_bg": "#0f172a",
        "listbox_fg": "#e2e8f0",
        "listbox_select_bg": "#334155",
        "listbox_select_fg": "#e2e8f0",
        "hover_bg": "#1f2a44",
        "hover_fg": "#e2e8f0",
        "disabled_entry": "#111827",
        "focus_border": "#7c8799",
        "enabled_border": "#7c8799",
        "disabled_border": "#1e293b",
    },
}

DEFAULTS = {
    "Problem": {
        "type": "Expr",
        "u_exact_expr": "<manufactured_u0_expr>",
        "v_exact_expr": "<manufactured_v0_expr>",
        "f_exact_expr": "<manufactured_f_expr>",
        "u0_expr": "x*(1-x)*y*(1-y)",
        "v0_expr": "0",
        "f_expr": "0",
        "mu_expr": "1",
    },
    "Boundary condition": {
        "type": "Zero",
        "g_expr": "0",
        "v_expr": "0",
    },
    "Mesh": {
        "mesh_file": "mesh/square_structured.geo",
        "degree": "1",
    },
    "Time": {
        "T": "1.0",
        "dt": "0.01",
        "scheme": "Theta",
        "theta": "1.0",
        "beta": "0.25",
        "gamma": "0.50",
    },
    "Output": {
        "every": "1",
        "enable_progress_bar": True,
        "compute_error": False,
        "convergence_study": False,
        "convergence_type": "Time",
        "convergence_csv": "",
        "error_file": "build/error_history.csv",
        "vtk_directory": "build",
    },
}


# Predefined choice lists and help texts for specific parameters
CHOICES = {
    ("Problem", "type"): ["MMS", "Expr"],
    ("Boundary condition", "type"): ["Zero", "MMS", "Expr"],
    ("Time", "scheme"): ["Theta", "CentralDifference", "Newmark"],
    ("Output", "convergence_type"): ["Time", "Space"],
}

HELP = {
    # Problem
    ("Problem", "type"): "Choose problem type: 'MMS' uses the manufactured solution (useful for error checks); 'Expr' lets you provide expressions for initial/forcing terms.",
    ("Problem", "u_exact_expr"): "Exact initial displacement expression (used for MMS). Provide a function of x,y,t matching the manufactured solution at t=0.",
    ("Problem", "v_exact_expr"): "Exact initial velocity expression (used for MMS). Provide du_ex/dt at t=0 if using MMS.",
    ("Problem", "f_exact_expr"): "Exact forcing term (used for MMS). For MMS tests this is the analytic RHS matching the manufactured solution.",
    ("Problem", "u0_expr"): "Initial displacement expression for 'Expr' problems. Use variables x,y,t and math functions (e.g., 'sin(pi*x)').",
    ("Problem", "v0_expr"): "Initial velocity expression for 'Expr' problems. Use x,y,t if needed.",
    ("Problem", "f_expr"): "Forcing term expression for 'Expr' problems. Use 0 for no forcing.",
    ("Problem", "mu_expr"): "Coefficient mu(x,y,t) used in the PDE (default 1). You can enter spatially varying coefficients as an expression.",

    # Boundary condition
    ("Boundary condition", "type"): "Boundary type: 'Zero' imposes homogeneous Dirichlet; 'MMS' uses manufactured boundary values; 'Expr' uses the provided g_expr and v_expr expressions.",
    ("Boundary condition", "g_expr"): "Dirichlet boundary expression for displacement u(x,y,t). Required if boundary type is 'Expr'. If using MMS, this is taken from the manufactured solution (Problem.u_exact_expr), so no need to set manually.",
    ("Boundary condition", "v_expr"): "Dirichlet boundary expression for velocity v(x,y,t). Required if boundary type is 'Expr'. If using MMS, this is taken from the manufactured velocity (Problem.v_exact_expr), so no need to set manually.",

    # Mesh
    ("Mesh", "mesh_file"): "Path to the mesh file (.geo or .msh). Use the picker to select an existing mesh in the 'mesh' folder.",
    ("Mesh", "degree"): "Polynomial degree for the finite element basis (positive integer). Increase for higher-order accuracy.",

    # Time
    ("Time", "T"): "Final time of the simulation (real number). The simulation runs from t=0 to t=T.",
    ("Time", "dt"): "Time step size. Choose dt small enough for stability and accuracy (depends on scheme and mesh).",
    ("Time", "scheme"): "Time integration scheme: 'Theta' (general theta method), 'CentralDifference', or 'Newmark'.",
    ("Time", "theta"): "Theta parameter for the theta scheme: 0 explicit, 0.5 Crank-Nicolson, 1 implicit/backward Euler. Used only when Time.scheme='Theta'.",
    ("Time", "beta"): "Newmark-beta parameter (used only when Time.scheme='Newmark').",
    ("Time", "gamma"): "Newmark-gamma parameter (used only when Time.scheme='Newmark').",

    # Output
    ("Output", "every"): "Write VTK output every N time steps (integer). Set to 1 to write every step.",
    ("Output", "compute_error"): "When true (and problem type is MMS) the code computes and saves the error history to CSV.",
    ("Output", "convergence_study"): "Enable convergence study (ONLY for MMS). If enabled, you must choose a convergence type (time/space).",
    ("Output", "convergence_type"): "Convergence study type: 'Time' runs dt-refinement studies; 'Space' runs mesh-refinement studies. Used only if convergence_study is enabled.",
    ("Output", "convergence_csv"): "Optional path to a CSV file where the convergence table will be saved. Enabled only when convergence_study is true (and Problem.type is MMS). Leave empty to disable CSV output.",
    ("Output", "error_file"): "Path to the CSV file where the error history will be saved. Enabled only if compute_error is true.",
    ("Output", "vtk_directory"): "Directory where VTK (.vtu/.pvtu) output files will be written. Use the picker to choose an output folder.",
    ("Output", "enable_progress_bar"): "Whether to enable the progress bar during time-stepping. It will be shown only on the master MPI process. Set to false to disable it. If the output is not a TTY (e.g., redirected to a file), the progress bar will be disabled automatically.",
}

PATH_FIELDS = {
    ("Mesh", "mesh_file"),
    ("Output", "error_file"),
    ("Output", "vtk_directory"),
    ("Output", "convergence_csv"),
}


class ToolTip:
    def __init__(self, widget, text, delay=500, wrap=380):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wrap = wrap
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.overrideredirect(True)
        self._tip.attributes("-topmost", True)
        label = ttk.Label(self._tip, text=self.text, style="Tooltip.TLabel", wraplength=self.wrap, justify=tk.LEFT)
        label.pack(ipadx=8, ipady=6)
        self._tip.geometry(f"+{x}+{y}")

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def parse_prm_file(path: Path):
    """Robust parser for simple 'subsection NAME' / 'set key = value' files.

    - Case-insensitive for keywords `subsection` and `set`.
    - Strips surrounding quotes from values.
    - Removes trailing comments introduced with `#` or `//`.

    Supports nested sections by representing them as 'Parent/Child' in the returned dict.
    """
    data = {}
    section_stack = []
    set_re = re.compile(r"^\s*set\s+(\S+)\s*=\s*(.*)$", flags=re.IGNORECASE)
    sub_re = re.compile(r"^\s*subsection\s+(.+)$", flags=re.IGNORECASE)

    def current_section_name():
        return "/".join(section_stack) if section_stack else None

    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # remove inline comments (simple heuristic)
            for delim in ("#", "//"):
                if delim in line:
                    line = line.split(delim, 1)[0].strip()
            if not line:
                continue

            m = sub_re.match(line)
            if m:
                sec = m.group(1).strip()
                # strip surrounding quotes from section name
                if (sec.startswith('"') and sec.endswith('"')) or (
                        sec.startswith("'") and sec.endswith("'")):
                    sec = sec[1:-1]
                section_stack.append(sec)
                full = current_section_name()
                if full not in data:
                    data[full] = {}
                continue

            if line.lower() == "end":
                if section_stack:
                    section_stack.pop()
                continue

            m = set_re.match(line)
            sec_name = current_section_name()
            if m and sec_name is not None:
                key = m.group(1).strip()
                val = m.group(2).strip()
                # strip possible surrounding quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                # convert booleans
                if isinstance(val, str) and val.lower() in ("true", "false"):
                    val = val.lower() == "true"
                data.setdefault(sec_name, {})[key] = val

    return data


class PrmGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x640")
        self.minsize(860, 560)
        self.theme_name = "light"
        self.style = ttk.Style(self)
        # store mapping: section -> key -> {var: tk.Variable, widget: widget}
        self.vars = {}
        self._tooltip_instances = []

        self.apply_theme(self.theme_name)
        self._build_menu()

        self._build_ui()
        self.load_defaults()

    def _build_menu(self):
        menubar = tk.Menu(self)
        view = tk.Menu(menubar, tearoff=0)
        view.add_command(label="Light theme", command=lambda: self.apply_theme("light"))
        view.add_command(label="Dark theme", command=lambda: self.apply_theme("dark"))
        menubar.add_cascade(label="View", menu=view)
        self.config(menu=menubar)
        self._menubar = menubar
        self._view_menu = view

    def apply_theme(self, name: str):
        theme = THEMES.get(name, THEMES["light"])
        self.theme_name = name
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.configure(bg=theme["bg"])
        self.style.configure("App.TFrame", background=theme["bg"])
        self.style.configure("Panel.TFrame", background=theme["panel"], borderwidth=1, relief="solid")
        self.style.configure("Header.TLabel", background=theme["bg"], foreground=theme["text"], font=("Segoe UI", 12, "bold"))
        self.style.configure("Muted.TLabel", background=theme["bg"], foreground=theme["muted"], font=("Segoe UI", 9))
        self.style.configure("FieldReason.TLabel", background=theme["panel"], foreground=theme["text"], font=("Segoe UI", 9, "italic"))
        self.style.configure("Field.TLabel", background=theme["panel"], foreground=theme["text"], font=("Segoe UI", 10))
        self.style.configure("FieldDisabled.TLabel", background=theme["panel"], foreground=theme["muted"], font=("Segoe UI", 10))
        self.style.configure("TLabel", background=theme["panel"], foreground=theme["text"], font=("Segoe UI", 10))
        self.style.configure("TFrame", background=theme["panel"])
        self.style.configure("TNotebook", background=theme["bg"], borderwidth=0)
        self.style.configure("TNotebook.Tab", padding=(16, 6), font=("Segoe UI", 10))
        self.style.map("TNotebook.Tab", background=[("selected", theme["panel"]), ("!selected", theme["bg"])],
                       foreground=[("selected", theme["text"]), ("!selected", theme["muted"])])

        self.style.configure(
            "TEntry",
            fieldbackground=theme["entry"],
            foreground=theme["text"],
            bordercolor=theme["border"],
            lightcolor=theme["border"],
            darkcolor=theme["border"],
            borderwidth=1,
            selectbackground=theme["select_bg"],
            selectforeground=theme["select_fg"],
        )
        self.style.map(
            "TEntry",
            fieldbackground=[("disabled", theme["disabled_entry"])],
            foreground=[("disabled", theme["muted"])],
            bordercolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
            lightcolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
            darkcolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
        )
        self.style.configure(
            "TCombobox",
            fieldbackground=theme["entry"],
            foreground=theme["text"],
            arrowcolor=theme["text"],
            bordercolor=theme["border"],
            lightcolor=theme["border"],
            darkcolor=theme["border"],
            borderwidth=1,
            selectbackground=theme["select_bg"],
            selectforeground=theme["select_fg"],
        )
        self.style.configure(
            "TCheckbutton",
            background=theme["panel"],
            foreground=theme["text"],
            indicatorbackground=theme["entry"],
            indicatorforeground=theme["text"],
        )
        self.style.configure("Inline.TCheckbutton", padding=(6, 4))
        self.style.map(
            "TCheckbutton",
            background=[("active", theme.get("hover_bg", theme["panel"]))],
            foreground=[("active", theme.get("hover_fg", theme["text"]))],
        )
        self.style.configure("TButton", padding=(10, 6))
        self.style.map(
            "TButton",
            background=[("active", theme.get("hover_bg", theme["panel"]))],
            foreground=[("active", theme.get("hover_fg", theme["text"]))],
        )
        self.style.configure("Primary.TButton", background=theme["accent"], foreground="white", bordercolor=theme["accent_dark"], focusthickness=0)
        self.style.map(
            "Primary.TButton",
            background=[("active", theme["accent_dark"]), ("disabled", theme["border"])],
            foreground=[("disabled", theme["muted"])],
        )
        self.style.configure("Secondary.TButton", background=theme["panel"], foreground=theme["text"], bordercolor=theme["border"], focusthickness=0)
        self.style.map(
            "Secondary.TButton",
            background=[("active", theme.get("hover_bg", theme["panel"])), ("disabled", theme["panel"])],
            foreground=[("active", theme.get("hover_fg", theme["text"])), ("disabled", theme["muted"])],
        )
        self.style.configure("Inline.TButton", padding=(6, 4), font=("Segoe UI", 9), background=theme["panel"], foreground=theme["text"], bordercolor=theme["border"], focusthickness=0)
        self.style.map(
            "Inline.TButton",
            background=[("active", theme.get("hover_bg", theme["panel"])), ("disabled", theme["panel"])],
            foreground=[("active", theme.get("hover_fg", theme["text"])), ("disabled", theme["muted"])],
        )
        self.style.configure("Help.TButton", padding=(4, 2), font=("Segoe UI", 8), background=theme["panel"], foreground=theme["text"], bordercolor=theme["border"], focusthickness=0)
        self.style.map(
            "Help.TButton",
            background=[("active", theme.get("hover_bg", theme["panel"])), ("disabled", theme["panel"])],
            foreground=[("active", theme.get("hover_fg", theme["text"])), ("disabled", theme["muted"])],
            bordercolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
            lightcolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
            darkcolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
        )
        self.style.configure("Tooltip.TLabel", background=theme["panel"], foreground=theme["text"], relief="solid", borderwidth=1)

        self.option_add("*Font", ("Segoe UI", 10))
        self.option_add("*Background", theme["bg"])
        self.option_add("*Foreground", theme["text"])
        self.option_add("*TCombobox*Listbox*background", theme["listbox_bg"])
        self.option_add("*TCombobox*Listbox*foreground", theme["listbox_fg"])
        self.option_add("*TCombobox*Listbox*selectBackground", theme["listbox_select_bg"])
        self.option_add("*TCombobox*Listbox*selectForeground", theme["listbox_select_fg"])
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", theme["entry"]), ("active", theme["entry"]), ("focus", theme["entry"]), ("disabled", theme["disabled_entry"])],
            foreground=[("readonly", theme["text"]), ("active", theme["text"]), ("focus", theme["text"]), ("disabled", theme["muted"])],
            background=[("readonly", theme["entry"]), ("active", theme["entry"]), ("focus", theme["entry"]), ("disabled", theme["disabled_entry"])],
            arrowcolor=[("active", theme["text"]), ("focus", theme["text"]), ("disabled", theme["muted"]), ("!disabled", theme["text"])],
            bordercolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
            lightcolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
            darkcolor=[("disabled", theme["disabled_border"]), ("!disabled", theme["enabled_border"])],
        )

        self._apply_boolean_theme()

        if hasattr(self, "_menubar"):
            self._menubar.configure(background=theme["bg"], foreground=theme["text"], activebackground=theme.get("hover_bg", theme["panel"]), activeforeground=theme["text"], borderwidth=0)
        if hasattr(self, "_view_menu"):
            self._view_menu.configure(background=theme["panel"], foreground=theme["text"], activebackground=theme.get("hover_bg", theme["panel"]), activeforeground=theme["text"], borderwidth=0)

    def _build_ui(self):
        outer = ttk.Frame(self, style="App.TFrame")
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill=tk.X, padx=14, pady=(12, 6))
        ttk.Label(header, text=APP_TITLE, style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="Create and edit deal.II .prm files", style="Muted.TLabel").pack(side=tk.LEFT, padx=12)

        # left: notebook with sections
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=8)

        for section, entries in DEFAULTS.items():
            f = ttk.Frame(self.notebook, style="Panel.TFrame", padding=(10, 8))
            self.notebook.add(f, text=section)
            self.vars[section] = {}
            f.columnconfigure(1, weight=1)
            f.columnconfigure(4, weight=1)
            row = 0
            for key, default in entries.items():
                pick_btn = None
                pretty_key = key.replace("_", " ").title()
                lbl = ttk.Label(f, text=pretty_key, style="Field.TLabel")
                lbl.grid(row=row, column=0, sticky=tk.W, padx=(6, 8), pady=6)
                info_btn = None
                # Provide a combobox for known enumerations
                if (section, key) in CHOICES:
                    values = CHOICES[(section, key)]
                    var = tk.StringVar(value=str(default))
                    widget = ttk.Combobox(f, textvariable=var, values=values, state="readonly")
                    widget.grid(row=row, column=1, sticky=tk.EW, padx=(0, 6))
                # Provide path picker for path-like fields
                elif (section, key) in PATH_FIELDS:
                    var = tk.StringVar(value=str(default))
                    widget = ttk.Entry(f, textvariable=var)
                    widget.grid(row=row, column=1, sticky=tk.EW, padx=(0, 6))
                    def make_picker(s=section, k=key, v=var):
                        def _pick():
                            if (s, k) == ("Mesh", "mesh_file"):
                                p = filedialog.askopenfilename(title="Select mesh file", filetypes=[("Mesh files", ("*.geo", "*.msh", "*.*"))])
                                if p:
                                    v.set(p)
                            elif (s, k) == ("Output", "error_file"):
                                p = filedialog.asksaveasfilename(title="Select error CSV file", defaultextension=".csv", filetypes=[("CSV files", ("*.csv", "*.*"))])
                                if p:
                                    v.set(p)
                            elif (s, k) == ("Output", "vtk_directory"):
                                p = filedialog.askdirectory(title="Select VTK output directory")
                                if p:
                                    v.set(p)
                            elif (s, k) == ("Output", "convergence_csv"):
                                p = filedialog.asksaveasfilename(title="Select convergence CSV file", defaultextension=".csv", filetypes=[("CSV files", ("*.csv", "*.*"))])
                                if p:
                                    v.set(p)
                        return _pick
                    pick_btn = ttk.Button(f, text="Browse", style="Help.TButton", command=make_picker())
                    pick_btn.grid(row=row, column=2, sticky=tk.W, padx=(0, 6))
                    info_btn = None
                else:
                    # boolean -> checkbox, else entry
                    if isinstance(default, bool):
                        var = tk.BooleanVar(value=default)
                        widget = tk.Checkbutton(f, variable=var, text="", takefocus=False, borderwidth=0, highlightthickness=0)
                        widget.grid(row=row, column=1, sticky=tk.W, padx=(0, 6))
                    else:
                        var = tk.StringVar(value=str(default))
                        widget = ttk.Entry(f, textvariable=var)
                        widget.grid(row=row, column=1, sticky=tk.EW, padx=(0, 6))

                # store both var and widget for dynamic enabling/disabling
                reason_lbl = ttk.Label(f, text="", style="FieldReason.TLabel")
                reason_lbl.grid(row=row, column=4, sticky=tk.W, padx=(0, 6))

                self.vars[section][key] = {
                    "var": var,
                    "widget": widget,
                    "reason_label": reason_lbl,
                    "pick_btn": locals().get("pick_btn"),
                    "label": lbl,
                    "is_bool": isinstance(default, bool),
                    "section": section,
                }

                # attach watchers for dependent fields
                if section == "Problem" and key == "type":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Boundary condition" and key == "type":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Output" and key == "compute_error":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Output" and key == "convergence_study":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Time" and key == "scheme":
                    var.trace_add("write", lambda *a: self.update_widget_states())

                # add small info button when help text is available
                if (section, key) in HELP:
                    def make_info(text=HELP[(section, key)]):
                        return lambda: messagebox.showinfo("Help", text)
                    ib = ttk.Button(f, text="?", style="Help.TButton", width=2, command=make_info())
                    ib.grid(row=row, column=3, sticky=tk.W, padx=(0, 6))
                    self._tooltip_instances.append(ToolTip(widget, HELP[(section, key)]))

                row += 1

        # bottom controls
        ctrl = ttk.Frame(outer, style="App.TFrame")
        ctrl.pack(fill=tk.X, padx=12, pady=(4, 12))

        btn_load = ttk.Button(ctrl, text="Load .prm", style="Secondary.TButton", command=self.load_prm)
        btn_load.pack(side=tk.LEFT, padx=(0, 6))
        btn_defaults = ttk.Button(ctrl, text="Load defaults", style="Secondary.TButton", command=self.load_defaults)
        btn_defaults.pack(side=tk.LEFT, padx=(0, 6))
        btn_preview = ttk.Button(ctrl, text="Preview", style="Secondary.TButton", command=self.preview)
        btn_preview.pack(side=tk.LEFT, padx=(0, 6))
        btn_save = ttk.Button(ctrl, text="Save .prm", style="Primary.TButton", command=self.save_prm)
        btn_save.pack(side=tk.RIGHT)

        self._tooltip_instances.extend([
            ToolTip(btn_load, "Open an existing .prm file and populate the fields."),
            ToolTip(btn_defaults, "Restore the default parameters for all sections."),
            ToolTip(btn_preview, "Preview the generated .prm file before saving."),
            ToolTip(btn_save, "Validate and save the .prm file to disk."),
        ])

        self._apply_boolean_theme()

    def _apply_boolean_theme(self):
        if not hasattr(self, "vars"):
            return
        theme = THEMES.get(self.theme_name, THEMES["light"])
        for sec, kv in self.vars.items():
            for key, record in kv.items():
                if not record.get("is_bool"):
                    continue
                widget = record.get("widget")
                is_output = record.get("section") == "Output"
                font_size = 14 if is_output else 12
                pady = 4 if is_output else 2
                padx = 8 if is_output else 6
                try:
                    widget.configure(
                        bg=theme["panel"],
                        activebackground=theme.get("hover_bg", theme["panel"]),
                        fg=theme["text"],
                        activeforeground=theme["text"],
                        disabledforeground=theme["muted"],
                        selectcolor=theme["entry"],
                        font=("Segoe UI", font_size),
                        padx=padx,
                        pady=pady,
                    )
                except Exception:
                    pass

    def load_defaults(self):
        for s, entries in DEFAULTS.items():
            for k, v in entries.items():
                record = self.vars[s][k]
                var = record["var"]
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(v))
                else:
                    var.set(str(v))
        # apply dependency rules after loading defaults
        self.update_widget_states()
        messagebox.showinfo("Defaults", "Default values loaded.")

    def load_prm(self):
        p = filedialog.askopenfilename(
            title="Open .prm file",
            filetypes=[("PRM files", ("*.prm", "*.inp", "*.txt")), ("All files", "*.*")],
        )
        if not p:
            return
        try:
            data = parse_prm_file(Path(p))
        except Exception as e:
            messagebox.showerror("Error", f"Could not parse file: {e}")
            return
        # populate fields where possible
        for sec, kv in data.items():
            if sec not in self.vars:
                continue
            for k, v in kv.items():
                if k in self.vars[sec]:
                    record = self.vars[sec][k]
                    var = record["var"]
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(v))
                    else:
                        var.set(str(v))
        # apply dependency rules after loading a file
        self.update_widget_states()
        messagebox.showinfo("Loaded", f"Loaded values from {p}")

    def build_params_dict(self):
        out = {}
        for sec, kv in self.vars.items():
            out[sec] = {}
            for k, var in kv.items():
                v = var["var"]
                if isinstance(v, tk.BooleanVar):
                    out[sec][k] = v.get()
                else:
                    out[sec][k] = v.get()
        return out

    def format_prm(self, params: dict):
        lines = []

        # Special handling for Time: emit nested subsections for scheme parameters.
        for sec, kv in params.items():
            if sec != "Time":
                lines.append(f"subsection {sec}")
                for k, v in kv.items():
                    if isinstance(v, bool):
                        sval = "true" if v else "false"
                    else:
                        sval = v
                    lines.append(f"  set {k} = {sval}")
                lines.append("end")
                lines.append("")
                continue

            # Time section
            scheme = str(kv.get("scheme", "Theta")).strip()
            lines.append("subsection Time")
            for k in ("T", "dt", "scheme"):
                if k in kv:
                    lines.append(f"  set {k} = {kv[k]}")

            if scheme.lower() == "theta":
                lines.append("")
                lines.append("  subsection Theta")
                lines.append(f"    set theta = {kv.get('theta', '')}")
                lines.append("  end")
            elif scheme.lower() == "newmark":
                lines.append("")
                lines.append("  subsection Newmark")
                lines.append(f"    set beta = {kv.get('beta', '')}")
                lines.append(f"    set gamma = {kv.get('gamma', '')}")
                lines.append("  end")

            lines.append("end")
            lines.append("")

        return "\n".join(lines)

    def preview(self):
        params = self.build_params_dict()
        txt = self.format_prm(params)
        preview = tk.Toplevel(self)
        preview.title("Preview .prm")
        preview.geometry("820x520")
        preview.configure(bg=THEMES[self.theme_name]["preview_bg"])

        frame = ttk.Frame(preview, style="Panel.TFrame")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_bg = THEMES[self.theme_name]["preview_bg"]
        text_fg = THEMES[self.theme_name]["preview_fg"]

        t = tk.Text(frame, wrap=tk.NONE, bg=text_bg, fg=text_fg, insertbackground=text_fg,
                selectbackground=THEMES[self.theme_name]["select_bg"],
                selectforeground=THEMES[self.theme_name]["select_fg"],
                    font=("Consolas", 10), relief=tk.FLAT)
        t.insert("1.0", txt)
        t.config(state=tk.DISABLED)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=t.xview)
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=t.yview)
        t.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        t.grid(row=0, column=0, sticky=tk.NSEW)
        yscroll.grid(row=0, column=1, sticky=tk.NS)
        xscroll.grid(row=1, column=0, sticky=tk.EW)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def save_prm(self):
        p = filedialog.asksaveasfilename(
            title="Save .prm file",
            defaultextension=".prm",
            filetypes=[("PRM files", ("*.prm", "*.inp", "*.txt")), ("All files", "*.*")],
        )
        if not p:
            return
        params = self.build_params_dict()
        ok, msg = self.validate_before_save(params)
        if not ok:
            messagebox.showerror("Validation error", msg)
            return
        txt = self.format_prm(params)
        try:
            Path(p).write_text(txt)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write file: {e}")
            return
        messagebox.showinfo("Saved", f"Wrote {p}")

    def update_widget_states(self):
        """Enable/disable widgets based on dependency rules.

        Rules implemented:
        - Problem.type: 'MMS' -> enable exact_* fields, disable expr fields; 'Expr' -> enable expr fields, disable exact.
        - Boundary condition.type: 'Expr' -> enable g_expr and v_expr; otherwise disable them.
        - Time.scheme: enable theta only for Theta; enable beta/gamma only for Newmark; disable all three for CentralDifference.
        - Output.compute_error: when false -> disable error_file, when true -> enable.
        - Output.convergence_study: only enabled when Problem.type == 'MMS'; when enabled -> enable convergence_type, else disable.
        - Output.convergence_csv: only enabled when Problem.type == 'MMS' and Output.convergence_study is true.
        """
        # Problem.type rules
        try:
            ptype = self.vars["Problem"]["type"]["var"].get().strip().lower()
        except Exception:
            ptype = "expr"

        exact_keys = ["u_exact_expr", "v_exact_expr", "f_exact_expr"]
        expr_keys = ["u0_expr", "v0_expr", "f_expr"]

        for k in exact_keys:
            widget = self.vars["Problem"][k]["widget"]
            reason_lbl = self.vars["Problem"][k]["reason_label"]
            try:
                if ptype == "mms":
                    widget.configure(state="normal")
                    reason_lbl.configure(text="")
                else:
                    widget.configure(state="disabled")
                    reason_lbl.configure(text="Enabled only for MMS")
            except Exception:
                pass

        for k in expr_keys:
            widget = self.vars["Problem"][k]["widget"]
            reason_lbl = self.vars["Problem"][k]["reason_label"]
            try:
                if ptype == "expr":
                    widget.configure(state="normal")
                    reason_lbl.configure(text="")
                else:
                    widget.configure(state="disabled")
                    reason_lbl.configure(text="Enabled only for Expr")
            except Exception:
                pass

        # Boundary condition rules
        try:
            btype = self.vars["Boundary condition"]["type"]["var"].get().strip().lower()
        except Exception:
            btype = "zero"

        for key in ("g_expr", "v_expr"):
            widget = self.vars["Boundary condition"][key]["widget"]
            reason_lbl = self.vars["Boundary condition"][key]["reason_label"]
            try:
                if btype == "expr":
                    widget.configure(state="normal")
                    reason_lbl.configure(text="")
                else:
                    widget.configure(state="disabled")
                    reason_lbl.configure(text="Enabled only for Expr")
            except Exception:
                pass

        # Output.compute_error rules
        try:
            compute = self.vars["Output"]["compute_error"]["var"].get()
        except Exception:
            compute = False
        err_widget = self.vars["Output"]["error_file"]["widget"]
        err_reason = self.vars["Output"]["error_file"]["reason_label"]
        try:
            if compute:
                err_widget.configure(state="normal")
                err_reason.configure(text="")
            else:
                err_widget.configure(state="disabled")
                err_reason.configure(text="Enable compute_error")
        except Exception:
            pass

        # Output.convergence_study rules
        try:
            conv_enabled = self.vars["Output"]["convergence_study"]["var"].get()
        except Exception:
            conv_enabled = False

        # convergence_study checkbox itself should only be interactive in MMS mode
        conv_chk = self.vars["Output"]["convergence_study"]["widget"]
        conv_reason = self.vars["Output"]["convergence_study"]["reason_label"]
        try:
            if ptype == "mms":
                conv_chk.configure(state="normal")
                conv_reason.configure(text="")
            else:
                conv_chk.configure(state="disabled")
                conv_reason.configure(text="Requires Problem.type = MMS")
                # also force it off when not MMS, to avoid saving misleading configs
                if conv_enabled:
                    self.vars["Output"]["convergence_study"]["var"].set(False)
                    conv_enabled = False
        except Exception:
            pass

        conv_type_widget = self.vars["Output"]["convergence_type"]["widget"]
        conv_type_reason = self.vars["Output"]["convergence_type"]["reason_label"]
        try:
            if ptype == "mms" and conv_enabled:
                conv_type_widget.configure(state="readonly")
                conv_type_reason.configure(text="")
            else:
                conv_type_widget.configure(state="disabled")
                if ptype != "mms":
                    conv_type_reason.configure(text="Requires Problem.type = MMS")
                else:
                    conv_type_reason.configure(text="Enable convergence_study")
        except Exception:
            pass

        # Output.convergence_csv rules
        try:
            conv_csv_widget = self.vars["Output"]["convergence_csv"]["widget"]
            conv_csv_reason = self.vars["Output"]["convergence_csv"]["reason_label"]
            if ptype == "mms" and conv_enabled:
                conv_csv_widget.configure(state="normal")
                conv_csv_reason.configure(text="")
            else:
                conv_csv_widget.configure(state="disabled")
                if ptype != "mms":
                    conv_csv_reason.configure(text="Requires Problem.type = MMS")
                else:
                    conv_csv_reason.configure(text="Enable convergence_study")
                # clear it when disabled to avoid populating it inadvertently
                if self.vars["Output"]["convergence_csv"]["var"].get():
                    self.vars["Output"]["convergence_csv"]["var"].set("")
        except Exception:
            pass

        # -------------------- Time.scheme rules --------------------
        try:
            tscheme = self.vars["Time"]["scheme"]["var"].get().strip().lower()
        except Exception:
            tscheme = "theta"

        def _set_enabled(section: str, key: str, enabled: bool, reason: str = ""):
            """Keep the row visible, but enable/disable the input widget."""
            if section not in self.vars or key not in self.vars[section]:
                return
            rec = self.vars[section][key]
            w = rec.get("widget")
            reason_lbl = rec.get("reason_label")

            try:
                if enabled:
                    # combobox uses readonly; entries use normal
                    if isinstance(w, ttk.Combobox):
                        w.configure(state="readonly")
                    else:
                        w.configure(state="normal")
                    if reason_lbl is not None:
                        reason_lbl.configure(text="")
                else:
                    w.configure(state="disabled")
                    if reason_lbl is not None:
                        reason_lbl.configure(text=reason)
            except Exception:
                pass

        is_theta   = (tscheme == "theta")
        is_newmark = (tscheme == "newmark")
        is_cd      = (tscheme == "centraldifference")

        _set_enabled("Time", "theta", enabled=is_theta, reason="Used only for Theta")
        _set_enabled("Time", "beta", enabled=is_newmark, reason="Used only for Newmark")
        _set_enabled("Time", "gamma", enabled=is_newmark, reason="Used only for Newmark")

        # Optional hint message on the main Time tab (reuse dt's reason label)
        try:
            hint_lbl = self.vars["Time"]["scheme"]["reason_label"]
            if is_cd:
                hint_lbl.configure(text="No scheme parameters for CentralDifference")
            else:
                hint_lbl.configure(text="")
        except Exception:
            pass

        # Keep path pickers in sync with their entry state
        for sec, kv in self.vars.items():
            for key, record in kv.items():
                try:
                    state = str(record["widget"].cget("state"))
                    enabled = state != "disabled"
                except Exception:
                    enabled = True

                if record.get("is_bool"):
                    try:
                        record["widget"].configure(selectcolor=THEMES[self.theme_name]["entry"] if enabled else THEMES[self.theme_name]["disabled_entry"])
                    except Exception:
                        pass

                lbl = record.get("label")
                if lbl is not None:
                    try:
                        lbl.configure(style="Field.TLabel" if enabled else "FieldDisabled.TLabel")
                    except Exception:
                        pass

                pick_btn = record.get("pick_btn")
                if pick_btn is None:
                    continue
                try:
                    if enabled:
                        pick_btn.grid()
                    else:
                        pick_btn.grid_remove()
                except Exception:
                    pass

    def validate_before_save(self, params: dict):
        """Return (ok, message). Validate required fields based on rules."""
        # Problem type
        ptype = params.get("Problem", {}).get("type", "Expr").strip().lower()
        if ptype == "MMS":
            for k in ("u_exact_expr", "v_exact_expr", "f_exact_expr"):
                if not params.get("Problem", {}).get(k):
                    return False, f"Missing required Problem parameter: {k} for MMS type"
        if ptype == "Expr":
            for k in ("u0_expr", "v0_expr", "f_expr"):
                if not params.get("Problem", {}).get(k):
                    return False, f"Missing required Problem parameter: {k} for Expr type"

        btype = params.get("Boundary condition", {}).get("type", "Zero").strip().lower()
        if btype == "Expr":
            for k in ("g_expr", "v_expr"):
                if not params.get("Boundary condition", {}).get(k):
                    return False, f"Missing required Boundary condition parameter: {k} for Expr type"

        if params.get("Output", {}).get("compute_error") in (True, "true", "True"):
            if not params.get("Output", {}).get("error_file"):
                return False, "Missing Output.error_file while compute_error is enabled"

        # convergence study validation
        conv = params.get("Output", {}).get("convergence_study")
        if conv in (True, "true", "True"):
            if ptype != "MMS":
                return False, "Output.convergence_study can only be enabled when Problem.type is 'MMS'"
            ctype = str(params.get("Output", {}).get("convergence_type", "")).strip().lower()
            if ctype not in ("Time", "Space"):
                return False, "Missing/invalid Output.convergence_type (must be 'Time' or 'Space') when convergence_study is enabled"
        else:
            # if convergence is disabled, the csv path must not be set
            if params.get("Output", {}).get("convergence_csv"):
                return False, "Output.convergence_csv can only be set when Output.convergence_study is enabled"

        return True, ""


def main():
    app = PrmGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
