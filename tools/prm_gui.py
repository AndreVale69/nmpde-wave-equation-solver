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

DEFAULTS = {
    "Problem": {
        "type": "physical",
        "u0_exact_expr": "<manufactured_u0_expr>",
        "v0_exact_expr": "<manufactured_v0_expr>",
        "f_exact_expr": "<manufactured_f_expr>",
        "u0_expr": "x*(1-x)*y*(1-y)",
        "v0_expr": "0",
        "f_expr": "0",
        "mu_expr": "1",
    },
    "Boundary condition": {
        "type": "zero",
        "g_expr": "0",
        "v_expr": "0",
    },
    "Mesh": {
        "mesh_file": "../mesh/square_structured.geo",
        "degree": "1",
    },
    "Time": {
        "T": "1.0",
        "dt": "0.01",
        "theta": "1.0",
        "scheme": "theta",
    },
    "Output": {
        "every": "1",
        "compute_error": False,
        "error_file": "build/error_history.csv",
        "vtk_directory": "build",
    },
}


# Predefined choice lists and help texts for specific parameters
CHOICES = {
    ("Problem", "type"): ["physical", "mms", "expr"],
    ("Boundary condition", "type"): ["zero", "mms", "expr"],
    ("Time", "scheme"): ["theta", "centraldifference", "newmark"],
}

HELP = {
    # Problem
    ("Problem", "type"): "Choose problem type: 'physical' runs the physical problem; 'mms' uses the manufactured solution (useful for error checks); 'expr' lets you provide expressions for initial/forcing terms.",
    ("Problem", "u0_exact_expr"): "Exact initial displacement expression (used for MMS). Provide a function of x,y,t matching the manufactured solution at t=0.",
    ("Problem", "v0_exact_expr"): "Exact initial velocity expression (used for MMS). Provide du_ex/dt at t=0 if using MMS.",
    ("Problem", "f_exact_expr"): "Exact forcing term (used for MMS). For MMS tests this is the analytic RHS matching the manufactured solution.",
    ("Problem", "u0_expr"): "Initial displacement expression for 'expr' problems. Use variables x,y,t and math functions (e.g., 'sin(pi*x)').",
    ("Problem", "v0_expr"): "Initial velocity expression for 'expr' problems. Use x,y,t if needed.",
    ("Problem", "f_expr"): "Forcing term expression for 'expr' problems. Use 0 for no forcing.",
    ("Problem", "mu_expr"): "Coefficient mu(x,y,t) used in the PDE (default 1). You can enter spatially varying coefficients as an expression.",

    # Boundary condition
    ("Boundary condition", "type"): "Boundary type: 'zero' imposes homogeneous Dirichlet; 'mms' uses manufactured boundary values; 'expr' uses the provided g_expr and v_expr expressions.",
    ("Boundary condition", "g_expr"): "Dirichlet boundary expression for displacement u(x,y,t). Required if boundary type is 'expr'.",
    ("Boundary condition", "v_expr"): "Dirichlet boundary expression for velocity v(x,y,t). Required if boundary type is 'expr'.",

    # Mesh
    ("Mesh", "mesh_file"): "Path to the mesh file (.geo or .msh). Use the picker to select an existing mesh in the 'mesh' folder.",
    ("Mesh", "degree"): "Polynomial degree for the finite element basis (positive integer). Increase for higher-order accuracy.",

    # Time
    ("Time", "T"): "Final time of the simulation (real number). The simulation runs from t=0 to t=T.",
    ("Time", "dt"): "Time step size. Choose dt small enough for stability and accuracy (depends on scheme and mesh).",
    ("Time", "theta"): "Theta parameter for the theta scheme: 0 explicit, 0.5 Crank-Nicolson, 1 implicit/backward Euler.",
    ("Time", "scheme"): "Time integration scheme: 'theta' (general theta method), 'centraldifference', or 'newmark'.",

    # Output
    ("Output", "every"): "Write VTK output every N time steps (integer). Set to 1 to write every step.",
    ("Output", "compute_error"): "When true (and problem type is MMS) the code computes and saves the error history to CSV.",
    ("Output", "error_file"): "Path to the CSV file where the error history will be saved. Enabled only if compute_error is true.",
    ("Output", "vtk_directory"): "Directory where VTK (.vtu/.pvtu) output files will be written. Use the picker to choose an output folder.",
}

PATH_FIELDS = {
    ("Mesh", "mesh_file"),
    ("Output", "error_file"),
    ("Output", "vtk_directory"),
}


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
        self.title("PRM Editor")
        self.geometry("900x600")
        # store mapping: section -> key -> {var: tk.Variable, widget: widget}
        self.vars = {}

        self._build_ui()
        self.load_defaults()

    def _build_ui(self):
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True)

        # left: notebook with sections
        self.notebook = ttk.Notebook(frm)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        for section, entries in DEFAULTS.items():
            f = ttk.Frame(self.notebook)
            self.notebook.add(f, text=section)
            self.vars[section] = {}
            row = 0
            for key, default in entries.items():
                ttk.Label(f, text=key).grid(row=row, column=0, sticky=tk.W, padx=6, pady=4)
                info_btn = None
                # Provide a combobox for known enumerations
                if (section, key) in CHOICES:
                    values = CHOICES[(section, key)]
                    var = tk.StringVar(value=str(default))
                    widget = ttk.Combobox(f, textvariable=var, values=values, state="readonly")
                    widget.grid(row=row, column=1, sticky=tk.W, padx=6)
                # Provide path picker for path-like fields
                elif (section, key) in PATH_FIELDS:
                    var = tk.StringVar(value=str(default))
                    widget = ttk.Entry(f, textvariable=var, width=68)
                    widget.grid(row=row, column=1, sticky=tk.W, padx=(6, 0))
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
                        return _pick
                    pick_btn = ttk.Button(f, text="...", width=3, command=make_picker())
                    pick_btn.grid(row=row, column=2, sticky=tk.W, padx=4)
                    info_btn = None
                else:
                    # boolean -> checkbox, else entry
                    if isinstance(default, bool):
                        var = tk.BooleanVar(value=default)
                        widget = ttk.Checkbutton(f, variable=var)
                        widget.grid(row=row, column=1, sticky=tk.W, padx=6)
                    else:
                        var = tk.StringVar(value=str(default))
                        widget = ttk.Entry(f, textvariable=var, width=80)
                        widget.grid(row=row, column=1, sticky=tk.W, padx=6)

                # store both var and widget for dynamic enabling/disabling
                self.vars[section][key] = {"var": var, "widget": widget}

                # attach watchers for dependent fields
                if section == "Problem" and key == "type":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Boundary condition" and key == "type":
                    var.trace_add("write", lambda *a: self.update_widget_states())
                if section == "Output" and key == "compute_error":
                    var.trace_add("write", lambda *a: self.update_widget_states())

                # add small info button when help text is available
                if (section, key) in HELP:
                    def make_info(text=HELP[(section, key)]):
                        return lambda: messagebox.showinfo("Help", text)
                    ib = ttk.Button(f, text="i", width=2, command=make_info())
                    ib.grid(row=row, column=3, sticky=tk.W, padx=4)

                row += 1

        # bottom controls
        ctrl = ttk.Frame(self)
        ctrl.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(ctrl, text="Load .prm", command=self.load_prm).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Load defaults", command=self.load_defaults).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Preview", command=self.preview).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Save .prm", command=self.save_prm).pack(side=tk.RIGHT, padx=4)

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
        t = tk.Text(preview, wrap=tk.NONE)
        t.insert("1.0", txt)
        t.config(state=tk.DISABLED)
        t.pack(fill=tk.BOTH, expand=True)

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
        - Problem.type: 'mms' -> enable exact_* fields, disable expr fields; 'expr' -> enable expr fields, disable exact; 'physical' -> disable both groups.
        - Boundary condition.type: 'expr' -> enable g_expr and v_expr; otherwise disable them.
        - Output.compute_error: when false -> disable error_file, when true -> enable.
        """
        # Problem.type rules
        try:
            ptype = self.vars["Problem"]["type"]["var"].get().strip().lower()
        except Exception:
            ptype = "physical"

        exact_keys = ["u0_exact_expr", "v0_exact_expr", "f_exact_expr"]
        expr_keys = ["u0_expr", "v0_expr", "f_expr"]

        for k in exact_keys:
            widget = self.vars["Problem"][k]["widget"]
            try:
                if ptype == "mms":
                    widget.configure(state="normal")
                else:
                    widget.configure(state="disabled")
            except Exception:
                pass

        for k in expr_keys:
            widget = self.vars["Problem"][k]["widget"]
            try:
                if ptype == "expr":
                    widget.configure(state="normal")
                else:
                    widget.configure(state="disabled")
            except Exception:
                pass

        # Boundary condition rules
        try:
            btype = self.vars["Boundary condition"]["type"]["var"].get().strip().lower()
        except Exception:
            btype = "zero"

        for key in ("g_expr", "v_expr"):
            widget = self.vars["Boundary condition"][key]["widget"]
            try:
                if btype == "expr":
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

    def validate_before_save(self, params: dict):
        """Return (ok, message). Validate required fields based on rules."""
        # Problem type
        ptype = params.get("Problem", {}).get("type", "physical").strip().lower()
        if ptype == "mms":
            for k in ("u0_exact_expr", "v0_exact_expr", "f_exact_expr"):
                if not params.get("Problem", {}).get(k):
                    return False, f"Missing required Problem parameter: {k} for MMS type"
        if ptype == "expr":
            for k in ("u0_expr", "v0_expr", "f_expr"):
                if not params.get("Problem", {}).get(k):
                    return False, f"Missing required Problem parameter: {k} for Expr type"

        btype = params.get("Boundary condition", {}).get("type", "zero").strip().lower()
        if btype == "expr":
            for k in ("g_expr", "v_expr"):
                if not params.get("Boundary condition", {}).get(k):
                    return False, f"Missing required Boundary condition parameter: {k} for Expr type"

        if params.get("Output", {}).get("compute_error") in (True, "true", "True"):
            if not params.get("Output", {}).get("error_file"):
                return False, "Missing Output.error_file while compute_error is enabled"

        return True, ""


def main():
    app = PrmGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
