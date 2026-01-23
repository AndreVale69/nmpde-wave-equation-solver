def modal_explanation_text() -> str:
    return r"""
### What does this plot indicate?

This plot shows **how one vibration of the system evolves over time**.

- **Modal amplitude `a(t)`**  
  Tells *how strong* the vibration is.

- **Modal velocity `adot(t)`**  
  Tells *how fast* the vibration is moving.

Think of a **mass on a spring**:
- `a(t)` = position of the mass  
- `adot(t)` = speed of the mass

---

### How to read the curves (for dummies)

✅ **Good behavior**
  - Smooth oscillations;
  - Peaks stay roughly the same height.

The numerical method is **stable** and **accurate**.

⚠️ **Numerical damping**
  - Peaks slowly shrink over time $\to$ the method is losing energy (often acceptable, but noticeable).

❌ **Problematic behavior**
  - Peaks grow over time $\to$ instability (something is wrong).
  - Very jagged or noisy curves $\to$ time step is too large.

In summary, if the curves oscillate smoothly and do not grow or collapse too fast,
the numerical scheme behaves well for this vibration mode.
"""

def modal_explanation_for_dummies() -> str:
    return r"""
### Modal amplitude and velocity for dummies

#### 1. Big picture: what is a "mode" in this context?

In the wave equation, the solution is a **vibrating shape** over space.

Rather than examining the entire shape of the solution, $u_h(x, t)$,
we study **a specific vibration pattern**, called *mode*.
It's like saying:

> "How much is the solution vibrating **in this particular shape** over time?"

That single number changing in time is the **modal amplitude**.

So, the **modal analysis** focuses on tracking one vibration component over time.

---

#### 2. What is **modal amplitude** `a(t)`?

**`a(t)` answers this question:** _"How strong is this vibration mode at time `t`?"_.

In other words, `a(t)` tells us: at time `t`, how much is the solution vibrating in this specific shape?

Interpret it as:

* positive / negative $\to$ direction of vibration
* large magnitude $\to$ strong vibration
* small magnitude $\to$ weak vibration

##### How to read `a(t)` in the graph

* Oscillating curve $\to$ wave-like behavior (expected)
* Constant peak height $\to$ no damping
* Slowly shrinking peaks $\to$ damping (numerical or physical)
* Growing peaks $\to$ instability (bad sign)

In short, **if `a(t)` looks like a sine wave, the mode behaves correctly**.

---

#### 3. What is **modal velocity** `adot(t)`?

**`adot(t)` is just the time derivative of `a(t)`**.

In plain words:

> If `a(t)` tells us *where* the vibration is,
> `adot(t)` tells us *how fast it is moving*.

Analogy:

* `a(t)` $\to$ position of a mass on a spring
* `adot(t)` $\to$ its velocity

##### How to read `adot(t)`

* Peaks of `adot` occur when `a(t)` crosses zero
* Zeros of `adot` occur when `a(t)` is at max/min
* This phase shift is **normal and expected**

If `adot(t)` looks noisy or distorted, the time step is probably too large.

---

#### 4. How to read the **combined graph** (`a` and `adot` together)

When we plot both:
  - $a(t) \to$ oscillation
  - $\dot{a}(t) \to$ same oscillation, shifted in time

We should expect: same frequency, smooth curves, and no sudden jumps.

##### Typical interpretations

| What we see                            | Meaning                           |
| -------------------------------------- | --------------------------------- |
| Clean oscillations, constant amplitude | Good scheme, no numerical damping |
| Slowly shrinking amplitude             | Numerical damping                 |
| Rapid decay                            | Too much numerical dissipation    |
| Growing amplitude                      | Instability                       |
| Jagged / noisy `adot`                  | Time step too large               |

---

#### 5. Why is modal analysis useful?

Because **global error norms don't tell the full story**. Modal analysis reveals:

* numerical damping (even when errors are small),
* phase errors,
* long-time behavior problems.

It’s especially useful to compare explicit vs implicit schemes, different $\theta$ values, and time step sizes.

---

In summary, **modal amplitude tells us how strongly one vibration mode oscillates;
modal velocity tells us how fast it moves**. If both oscillate smoothly with stable amplitude, the method behaves well.
"""