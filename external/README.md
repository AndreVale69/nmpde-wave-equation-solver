## 📦 External Dependencies

This directory contains **third-party libraries** that are **not provided by the system environment**
or by the version of deal.II available on the target platform.

All dependencies placed here are:

* **Header-only** or lightweight
* **Vendored explicitly** to ensure reproducibility
* **Not modified**, unless explicitly stated

---

### 📚 Included libraries

#### `magic_enum` (v0.9.7)

* **Repository:** [https://github.com/Neargye/magic_enum](https://github.com/Neargye/magic_enum)
* **License:** MIT
* **Usage:** Compile-time enum reflection (enum $\iff$ string conversion)
* **Why it is vendored:**
  The version of deal.II used in this project (**9.5.1**) does **not** bundle `magic_enum`.
  Newer deal.II versions ($\ge$ 9.6) include it internally, but upgrading is not possible in the target HPC environment.

Only the official release headers are included:

```
external/magic_enum-v0.9.7/include/
```

No source files are compiled; the library is used as **header-only** via CMake `INTERFACE` targets.

---

### ⚠️ Modification policy

Do **not** edit files inside `external/` unless strictly necessary. If a modification is required:
  - Document it clearly in this README
  - Prefer upstream fixes when possible
