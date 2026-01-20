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
    },
    "dark": {
        "bg": "#0f172a",
        "panel": "#111827",
        "text": "#e5e7eb",
        "muted": "#9ca3af",
        "accent": "#60a5fa",
        "accent_dark": "#3b82f6",
        "border": "#1f2937",
        "entry": "#0b1220",
        "preview_bg": "#0b1220",
        "preview_fg": "#e5e7eb",
    },
}

DEFAULTS = {
    "Problem": {
        "type": "Physical",
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
        "theta": "1.0",
        "scheme": "Theta",
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
    ("Problem", "type"): ["Physical", "MMS", "Expr"],
    ("Boundary condition", "type"): ["Zero", "MMS", "Expr"],
    ("Time", "scheme"): ["Theta", "CentralDifference", "Newmark"],
    ("Output", "convergence_type"): ["Time", "Space"],
}

HELP = {
    # Problem
    ("Problem", "type"): "Choose problem type: 'Physical' runs the physical problem; 'MMS' uses the manufactured solution (useful for error checks); 'Expr' lets you provide expressions for initial/forcing terms.",
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
    ("Time", "theta"): "Theta parameter for the theta scheme: 0 explicit, 0.5 Crank-Nicolson, 1 implicit/backward Euler.",
    ("Time", "scheme"): "Time integration scheme: 'Theta' (general theta method), 'CentralDifference', or 'Newmark'.",

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
    """
    data = {}
    section = None
    set_re = re.compile(r"^\s*set\s+(\S+)\s*=\s*(.*)$", flags=re.IGNORECASE)
    sub_re = re.compile(r"^\s*subsection\s+(.+)$", flags=re.IGNORECASE)
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
                section = m.group(1).strip()
                # strip surrounding quotes from section name
                if (section.startswith('"') and section.endswith('"')) or (
                        section.startswith("'") and section.endswith("'")):
                    section = section[1:-1]
                data[section] = {}
                continue

            if line.lower() == "end":
                section = None
                continue

            m = set_re.match(line)
            if m and section is not None:
                key = m.group(1).strip()
                val = m.group(2).strip()
                # strip possible surrounding quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                # convert booleans
                if isinstance(val, str) and val.lower() in ("true", "false"):
                    val = val.lower() == "true"
                data[section][key] = val

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
        self.style.configure("Field.TLabel", background=theme["panel"], foreground=theme["text"], font=("Segoe UI", 10))
        self.style.configure("TLabel", background=theme["panel"], foreground=theme["text"], font=("Segoe UI", 10))
        self.style.configure("TFrame", background=theme["panel"])
        self.style.configure("TNotebook", background=theme["bg"], borderwidth=0)
        self.style.configure("TNotebook.Tab", padding=(16, 6), font=("Segoe UI", 10))
        self.style.map("TNotebook.Tab", background=[("selected", theme["panel"]), ("!selected", theme["bg"])],
                       foreground=[("selected", theme["text"]), ("!selected", theme["muted"])])

        self.style.configure("TEntry", fieldbackground=theme["entry"], foreground=theme["text"], bordercolor=theme["border"], lightcolor=theme["border"], darkcolor=theme["border"])
        self.style.configure("TCombobox", fieldbackground=theme["entry"], foreground=theme["text"], arrowcolor=theme["text"], bordercolor=theme["border"], lightcolor=theme["border"], darkcolor=theme["border"])
        self.style.configure("TCheckbutton", background=theme["panel"], foreground=theme["text"])
        self.style.configure("TButton", padding=(10, 6))
        self.style.configure("Primary.TButton", background=theme["accent"], foreground="white", bordercolor=theme["accent_dark"], focusthickness=0)
        self.style.map("Primary.TButton", background=[("active", theme["accent_dark"])])
        self.style.configure("Secondary.TButton", background=theme["panel"], foreground=theme["text"], bordercolor=theme["border"])
        self.style.configure("Tooltip.TLabel", background=theme["panel"], foreground=theme["text"], relief="solid", borderwidth=1)

        self.option_add("*Font", ("Segoe UI", 10))
        self.option_add("*Background", theme["bg"])
        self.option_add("*Foreground", theme["text"])

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
            row = 0
            for key, default in entries.items():
                ttk.Label(f, text=key, style="Field.TLabel").grid(row=row, column=0, sticky=tk.W, padx=(6, 8), pady=6)
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
                    pick_btn = ttk.Button(f, text="Browse", style="Secondary.TButton", command=make_picker())
                    pick_btn.grid(row=row, column=2, sticky=tk.W, padx=(0, 6))
                    info_btn = None
                else:
                    # boolean -> checkbox, else entry
                    if isinstance(default, bool):
                        var = tk.BooleanVar(value=default)
                        widget = ttk.Checkbutton(f, variable=var)
                        widget.grid(row=row, column=1, sticky=tk.W, padx=(0, 6))
                    else:
                        var = tk.StringVar(value=str(default))
                        widget = ttk.Entry(f, textvariable=var)
                        widget.grid(row=row, column=1, sticky=tk.EW, padx=(0, 6))

                # store both var and widget for dynamic enabling/disabling
                self.vars[section][key] = {"var": var, "widget": widget}

                # attach watchers for dependent fields
                if section == "Problem" and key == "type":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Boundary condition" and key == "type":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Output" and key == "compute_error":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Output" and key == "convergence_study":
                    var.trace_add("write", lambda *a: self.update_widget_states())

                # add small info button when help text is available
                if (section, key) in HELP:
                    def make_info(text=HELP[(section, key)]):
                        return lambda: messagebox.showinfo("Help", text)
                    ib = ttk.Button(f, text="?", style="Secondary.TButton", command=make_info())
                    ib.grid(row=row, column=3, sticky=tk.W, padx=(0, 6))
                    self._tooltip_instances.append(ToolTip(widget, HELP[(section, key)]))

                row += 1

        # bottom controls
        ctrl = ttk.Frame(outer, style="App.TFrame")
        ctrl.pack(fill=tk.X, padx=12, pady=(4, 12))

        ttk.Button(ctrl, text="Load .prm", style="Secondary.TButton", command=self.load_prm).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctrl, text="Load defaults", style="Secondary.TButton", command=self.load_defaults).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctrl, text="Preview", style="Secondary.TButton", command=self.preview).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(ctrl, text="Save .prm", style="Primary.TButton", command=self.save_prm).pack(side=tk.RIGHT)

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
        for sec, kv in params.items():
            lines.append(f"subsection {sec}")
            for k, v in kv.items():
                if isinstance(v, bool):
                    sval = "true" if v else "false"
                else:
                    sval = v
                lines.append(f"  set {k} = {sval}")
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
        - Problem.type: 'MMS' -> enable exact_* fields, disable expr fields; 'Expr' -> enable expr fields, disable exact; 'Physical' -> disable both groups.
        - Boundary condition.type: 'Expr' -> enable g_expr and v_expr; otherwise disable them.
        - Output.compute_error: when false -> disable error_file, when true -> enable.
        - Output.convergence_study: only enabled when Problem.type == 'MMS'; when enabled -> enable convergence_type, else disable.
        - Output.convergence_csv: only enabled when Problem.type == 'MMS' and Output.convergence_study is true.
        """
        # Problem.type rules
        try:
            ptype = self.vars["Problem"]["type"]["var"].get().strip().lower()
        except Exception:
            ptype = "Physical"

        exact_keys = ["u_exact_expr", "v_exact_expr", "f_exact_expr"]
        expr_keys = ["u0_expr", "v0_expr", "f_expr"]

        for k in exact_keys:
            widget = self.vars["Problem"][k]["widget"]
            try:
                if ptype == "MMS":
                    widget.configure(state="normal")
                else:
                    widget.configure(state="disabled")
            except Exception:
                pass

        for k in expr_keys:
            widget = self.vars["Problem"][k]["widget"]
            try:
                if ptype == "Expr":
                    widget.configure(state="normal")
                else:
                    widget.configure(state="disabled")
            except Exception:
                pass

        # Boundary condition rules
        try:
            btype = self.vars["Boundary condition"]["type"]["var"].get().strip().lower()
        except Exception:
            btype = "Zero"

        for key in ("g_expr", "v_expr"):
            widget = self.vars["Boundary condition"][key]["widget"]
            try:
                if btype == "Expr":
                    widget.configure(state="normal")
                else:
                    widget.configure(state="disabled")
            except Exception:
                pass

        # Output.compute_error rules
        try:
            compute = self.vars["Output"]["compute_error"]["var"].get()
        except Exception:
            compute = False
        err_widget = self.vars["Output"]["error_file"]["widget"]
        try:
            if compute:
                err_widget.configure(state="normal")
            else:
                err_widget.configure(state="disabled")
        except Exception:
            pass

        # Output.convergence_study rules
        try:
            conv_enabled = self.vars["Output"]["convergence_study"]["var"].get()
        except Exception:
            conv_enabled = False

        # convergence_study checkbox itself should only be interactive in MMS mode
        conv_chk = self.vars["Output"]["convergence_study"]["widget"]
        try:
            if ptype == "MMS":
                conv_chk.configure(state="normal")
            else:
                conv_chk.configure(state="disabled")
                # also force it off when not MMS, to avoid saving misleading configs
                if conv_enabled:
                    self.vars["Output"]["convergence_study"]["var"].set(False)
                    conv_enabled = False
        except Exception:
            pass

        conv_type_widget = self.vars["Output"]["convergence_type"]["widget"]
        try:
            if ptype == "MMS" and conv_enabled:
                conv_type_widget.configure(state="readonly")
            else:
                conv_type_widget.configure(state="disabled")
        except Exception:
            pass

        # Output.convergence_csv rules
        try:
            conv_csv_widget = self.vars["Output"]["convergence_csv"]["widget"]
            if ptype == "MMS" and conv_enabled:
                conv_csv_widget.configure(state="normal")
            else:
                conv_csv_widget.configure(state="disabled")
                # clear it when disabled to avoid populating it inadvertently
                if self.vars["Output"]["convergence_csv"]["var"].get():
                    self.vars["Output"]["convergence_csv"]["var"].set("")
        except Exception:
            pass

    def validate_before_save(self, params: dict):
        """Return (ok, message). Validate required fields based on rules."""
        # Problem type
        ptype = params.get("Problem", {}).get("type", "Physical").strip().lower()
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
