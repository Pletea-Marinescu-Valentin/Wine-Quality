# Worked Example: Class-Balanced Retrieval-Augmented Prompting

One complete end-to-end example from the evaluation pipeline. Model: **Gemma-3-12B** (Ollama `gemma3:12b`,
Q4_K_M, deterministic decoding: temperature 0.0, top-p 1.0, max 8 tokens).

## 1. Raw test instance

| Field | Value |
|---|---|
| Country | Italy |
| Region | Piedmont |
| Price | unknown |
| WineEnthusiast score | 94 |
| True class | 2 (High) |

**Review text:**

> This opens with a beautiful fragrance of rose, violet, perfumed berry, leather, baking spice and a balsamic note. The palate is still young, weaving together sour cherry, crushed raspberry, cinnamon, anise and mocha alongside assertive but finely grained tannins. Drink 2018–2024.

## 2. System prompt (identical across all strategies)

```text
You are a professional wine judge.
You must classify wines into exactly one class.

Wine quality classes (WineEnthusiast tiers):
0 = Low (80-82 points): flawed or very plain wines. Typical language: thin, vegetal, sour, bitter, green, dull, harsh, overripe, lacks.
1 = Medium (83-90 points): correct to very good everyday wines. Typical language: pleasant, balanced, easy, straightforward, crisp, clean, fruity, drink now.
2 = High (91-100 points): excellent to classic wines. Typical language: complex, concentrated, elegant, velvety, opulent, layered, refined, structured, minerality, long finish, ageworthy.

Statistical priors measured on thousands of expert reviews - use them as priors, the review text always has the final word:
- Base rates: about 2 percent of all wines are Low, 70 percent Medium and 28 percent High - when signals conflict, Medium is the safest call.
- Price ladder: under 10 USD is never High and often Low; 10-30 USD is mostly Medium; 30-50 USD is a Medium/High toss-up; 50-100 USD is likely High; over 100 USD is very likely High; over 300 USD is virtually always High. Unknown price leans Medium, sometimes High.
- Review length ladder (the length is given to you): under 190 characters is almost never High and carries real Low risk; 190-270 is usually Medium; 271-352 is an even Medium/High split; 353-433 is typically High; over 433 is almost always High.
- Strong combinations: a detailed review (over 350 chars) of a wine over 50 USD is High in 9 of 10 cases; a short review (under 190 chars) of a wine under 15 USD is never High and often Low.
- Variety priors: Nebbiolo, Gruener Veltliner, Pinot Noir, Champagne Blend, Syrah and Riesling lean High; Pinot Grigio, Rose, Portuguese White, White Blend, Sauvignon Blanc and Merlot rarely reach High.
- Origin priors: Austria and Germany lean High; Champagne, Piedmont, Burgundy, Mosel, Alsace, Douro and Oregon lean High; Chile, most Spanish and Argentine regions, and New York lean Low-to-Medium.
- The decisive boundary is Medium vs High (90 vs 91 points): High needs genuine superlatives, complexity/structure language or ageing potential; merely 'very good' is Medium.
- Low is rare (about 2 percent of wines): reserve it for clearly negative reviews.

CRITICAL RULES:
- Output ONLY one digit: 0, 1, or 2
- No explanation
- No punctuation
- No extra text
```

## 3. User prompt (class-balanced retrieval, K=6: top-2 exemplars per class by BGE cosine similarity)

```text
Similar wines for reference:

Country: Italy | Region: Piedmont | Price: 70 USD | Variety: Nebbiolo | Review length: 254 chars
Review: Intensely fragrant, this opens with scents of violet, rose, perfumed berry and a subtle whiff of menthol. The elegant, structured palate delivers tart cherry, white pepper, clove and a hint of coffee alongside firm, fine-grained tannins. Drink 2020–2028.
Class: 2

Country: Italy | Region: Piedmont | Price: unknown | Variety: Nebbiolo | Review length: 218 chars
Review: This opens with aromas of toast, espresso, cooking spice and red berry. The palate offers up young sour cherry, vanilla, coconut and mocha alongside still raspy tannins that leave an astringent finish. Drink 2017–2022.
Class: 1

Country: US | Region: Virginia | Price: 27 USD | Variety: Sangiovese | Review length: 213 chars
Review: Faint aromatics of bitter orange, rubber and anise biscotti unfold in the nose. Medium to full in body, this has structured tannins, with flavors that fade on the midpalate but resurface again on the brisk finish.
Class: 0

Country: Italy | Region: Piedmont | Price: 65 USD | Variety: Nebbiolo | Review length: 201 chars
Review: Aromas of leather, licorice, underbrush and chopped herb carry over to the palate together with tart red cherry, strawberry and star anise. Assertive but refined tannins offer support. Drink 2018–2024.
Class: 1

Country: US | Region: Texas | Price: 20 USD | Variety: Cabernet Sauvignon | Review length: 265 chars
Review: Aromas of raspberry and green peppercorn carry the nose. The palate is green, with underripe cranberry and wild blackberry accompanied by leather and white pepper. Underripe tannins are course throughout and dominate the finish with flavors of blackberry and cedar.
Class: 0

Country: Italy | Region: Piedmont | Price: 110 USD | Variety: Nebbiolo | Review length: 262 chars
Review: This opens with aromas of dark berry, forest floor, dried rose, new leather and a balsamic note. On the palate, notes of white pepper, coffee and clove accent a core of crunchy red berry. Racy acidity and assertive tannins provide the framework. Drink 2018–2023.
Class: 2

Now classify the final wine based on its own merits.

Country: Italy | Region: Piedmont | Price: unknown | Variety: Nebbiolo | Review length: 280 chars
Review: This opens with a beautiful fragrance of rose, violet, perfumed berry, leather, baking spice and a balsamic note. The palate is still young, weaving together sour cherry, crushed raspberry, cinnamon, anise and mocha alongside assertive but finely grained tannins. Drink 2018–2024.

Return ONLY one digit: 0, 1, or 2
```

## 4. Raw model output

```text
1
```

## 5. Parsed prediction

Predicted class: **1 (Medium)** — true class: **2 (High)**

The regex parser extracts the first standalone digit in {0, 1, 2} from the response.
