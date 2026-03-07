## RQ3: Do the recommendation scores make sense?

This section explains what we checked in Research Question 3 and what the
numbers mean, in plain language.

RQ3 asks two things:

- **Do the scores for each song really spread out, or are they all bunched
  together?**  
  If every song got almost the same score, the ranking would be arbitrary.

- **Does the scoring formula behave the way it is supposed to?**  
  The system’s score for a song combines:
  - how similar the song’s pitches are to the “ideal” profile for the singer
    (good), and
  - how much time the song spends on notes the singer wants to avoid (bad).

We want to see that:

- songs that look more like the ideal get **higher scores**, and
- songs that spend more time on “avoid” notes get **lower scores**,
- and that the code is really using the formula it claims to use.

All of these checks are done automatically from the pitch data; no human
judgments are involved.

---

### What data we used

- **Songs in the library**: 101 pieces (German Lieder and French mélodies)  
- **Synthetic “singer profiles”**: 25 different profiles
  - Each profile is built from one song’s pitch usage:
    - its most-used notes become “favourites”,
    - its least-used notes become “avoids”.
  - The profile also includes a vocal range (lowest and highest notes).
  - Only profiles that have **at least 10 matching songs** in the library
    are kept; this avoids degenerate cases with tiny candidate sets.
- **For each profile** we:
  - Filter the library to songs that fit the singer’s range.
  - Score all those songs using the same formula as the app.
  - Save the scores and some simple statistics.

So the RQ3 results are based on **25 different “simulated singers”**, each
with their own range and favourite/avoid notes, applied to the same library
of 101 songs.

---

### How we measure “spread” of scores

Every song in a profile’s candidate list gets a **final score** between 0 and 1:

- Higher = better match to the ideal profile,
- Lower = worse match.

For each profile (one run), we look at:

- **Variance of scores**: how much the scores differ from one another.
- **Range of scores**: the gap between the best-scoring and worst-scoring
  songs in that run.

Then we average these over the 25 profiles and compute a **95% confidence
interval (CI)**, which you can read as “a plausible range for the true
average”.

From the latest run:

- **Average variance of final scores**:  
  - Value: **0.040**  
  - 95% CI: **[0.035, 0.046]**

- **Average range of final scores** (difference between best and worst):  
  - Value: **0.758** (on a 0–1 scale)  
  - 95% CI: **[0.715, 0.798]**

#### How to read this

- A variance around **0.04** (on a 0–1 scale) means scores are not all
  clustered around the same value; there is a meaningful spread.
- A range around **0.76** means that, within a typical profile, the best
  and worst songs differ by roughly three-quarters of the full 0–1 scale.

In plain terms: **the system uses most of the score scale**, and songs are
spread out enough that the ranking is meaningful rather than arbitrary.

---

### Does the formula behave as advertised?

The system is supposed to score each song using:

- a **similarity** term (how close the song’s pitch usage is to the ideal),
  and
- a **penalty** for time spent on avoid notes,

combined in a simple formula:

> final score ≈ similarity − 0.5 × avoid_penalty

RQ3 checks this in two ways:

1. an **identity check** (does the code actually compute this formula?), and  
2. a small **regression check** (do we recover the expected weights 1 and −0.5?).

#### Identity check

For every song in every run, we:

- recompute `similarity − 0.5 × avoid_penalty`, and
- compare it to the stored final score.

We then look at:

- the **average absolute difference** between the two values, and
- the **maximum difference** seen in any song.

Results:

- **Average difference** across all runs: **0.0**  
- **Maximum difference** across all runs: **0.0**

So, up to numerical precision, the stored final scores are **exactly equal**
to “similarity minus 0.5 times the avoid time”, for every profile and song.
This is a very strong check that the implementation really matches the
described formula.

#### Regression check (sanity)

For each profile, we also run a simple statistical model where we try to
predict the final score from:

- the similarity, and
- the avoid penalty.

If the code is doing what it says, this model should discover:

- coefficient for similarity ≈ **+1**,  
- coefficient for avoid penalty ≈ **−0.5**, and  
- the model should explain essentially all the variation in scores.

Across the 25 profiles we found:

- **Average similarity weight**: **1.0**  
- **Average avoid-penalty weight**: **−0.5**  
- **Average “explained variance” (R²)**: **1.0**

In words: if you ask a simple statistical model to rediscover how scores are
computed, it says “add similarity with weight 1, subtract avoid time with
weight 0.5, and this perfectly explains all the scores”.

So: **the formula behaves exactly as designed**.

---

### Do the parts of the score point in the right direction?

Finally, we check whether the three ingredients behave in the way you would
expect, taken one pair at a time.

Across the 25 profiles, we summarise the relationships with **correlation
numbers** between −1 and 1:

- **+1**: as one goes up, the other goes up very consistently  
- **0**: no clear relationship  
- **−1**: as one goes up, the other goes down very consistently

Here’s what we see (averaged properly across profiles, with 95% CIs):

1. **Final score vs similarity** (good match should mean high score)

   - Average correlation: **0.988**  
   - 95% CI: **[0.984, 0.992]**

   Interpretation: songs that look more like the ideal profile **almost
   always** get higher scores. This is extremely close to a “perfect”
   relationship, which is exactly what we want given the formula.

2. **Final score vs avoid time** (more avoid notes should hurt the score)

   - Average correlation: **−0.466**  
   - 95% CI: **[−0.613, −0.285]**

   Interpretation: songs that spend more time on avoid notes tend to get
   **lower scores**, and this negative relationship is clearly different from
   zero. The size of the correlation is more modest than similarity’s
   effect, which matches the design choice that similarity is the main
   driver, with the avoid term acting as a correction.

3. **Similarity vs favourite-note overlap** (time on favourite notes)

   - Average correlation: **0.935**  
   - 95% CI: **[0.920, 0.946]**

   Interpretation: songs that spend more of their time on favourite notes
   are **very strongly** associated with higher similarity to the ideal,
   which is exactly what the ideal vector is meant to encode.

These are framed as **sanity checks** rather than deep “validity proofs”:
they confirm that the different parts of the score are pushing in the
intended directions and that nothing strange is happening inside the
pipeline.

---

### Plain-language conclusion for RQ3

Putting it together:

- **The scores spread out well**: within each simulated singer profile,
  songs cover most of the 0–1 range, so the ranking is not arbitrary.
- **The formula is implemented exactly as described**: “similarity minus
  0.5 times avoid time” holds to numerical precision, and a statistical
  model rediscovers the same formula with essentially perfect fit.
- **The components behave sensibly**:
  - more similarity → clearly higher scores,
  - more avoid-note time → clearly lower scores,
  - more time on favourite notes → much higher similarity to the ideal.

In short: **the RQ3 experiment supports that the scoring numbers are both
meaningful and faithfully computed from the intended formula**, in a way
that should be understandable to a reader without a strong maths or computer
science background.

