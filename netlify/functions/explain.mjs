// Netlify Function: explains a food product in plain words using Claude.
// Modes: product data (from a barcode lookup), a photo of the label (ingredients and/or nutrition panel),
// a receipt photo or pasted order email (what was bought), and a fridge photo (what's on the shelves).
// The API key stays here on the server (Netlify → Site configuration → Environment variables):
//   ANTHROPIC_API_KEY  (required)
//   ANTHROPIC_MODEL    (optional, default claude-haiku-5-5: fast and low-cost for short answers)
//   ANTHROPIC_VERSION  (optional, default 2023-06-01, the Messages API version header)
//   ALLOWED_ORIGINS    (optional, comma-separated, e.g. http://localhost:8501,https://your-site.netlify.app)
const MODEL = process.env.ANTHROPIC_MODEL || "claude-haiku-5-5";
const VERSION = process.env.ANTHROPIC_VERSION || "2023-06-01";

const SYSTEM = `You explain packaged food to everyday shoppers in plain, friendly words.
Rules:
- Use the product data you are given. Do not invent ingredients or numbers.
- Write for someone with no nutrition knowledge. No jargon; if you must name an additive, say what it is in a few words.
- Be calm and fair: mention real positives as well as concerns. No scare words, no medical advice, no diagnoses.
- If something conflicts with the household's diet or allergies, say so first.
- Keep it short: "summary" is at most 2 sentences (40 words). "points" has at most 3 items, each under 16 words.
- If the ingredient list is missing, say you can only judge from the nutrition numbers.
Reply with only JSON: {"summary": string, "points": [string]}`;

const LABEL = `You read photos of food packaging for everyday shoppers.
First, transcribe what is printed. Then explain it in plain, friendly words.
Rules:
- Copy the ingredient list and the nutrition numbers exactly as printed. Translate to English if needed. Never guess numbers that aren't visible: use null.
- Sodium in mg. If only salt is printed, sodium_mg = salt in g x 400.
- Allergens: list ones named or bolded on the label, as single lowercase words (milk, egg, peanut, tree_nut, soy, gluten, fish, shellfish, sesame).
- The explanation follows the same rules as for any product: calm, fair, no jargon, no scare words, no medical advice; diet or allergy conflicts with the household first.
- "summary" at most 2 sentences (40 words); "points" at most 3 items, each under 16 words.
- If the photo is not a food label or is unreadable, reply {"error":"unreadable"}.
Reply with only JSON:
{"name": string|null, "ingredients": string|null, "serving": string|null, "serving_g": number|null, "pack": string|null,
 "per_serving": {"kcal","fat","sat","carbs","sugar","fiber","protein","sodium_mg"}|null,
 "per_100g": {"kcal","fat","sat","carbs","sugar","fiber","protein","sodium_mg"}|null,
 "allergens": [string], "summary": string, "points": [string]}`;

const BOUGHT = `You turn grocery receipts and online order emails into a clean item list.
You get the household's product catalogue (id, name, package unit). Match each food line to the closest catalogue id.
Rules:
- qty is in CATALOGUE units: convert the size bought to the catalogue's package (catalogue "1 L" milk, bought 1 gallon -> qty 3.8; catalogue "12 count" eggs, bought 18 -> 1.5). If no size is shown, qty = number of packages.
- price is the line total in dollars, or null.
- Lines that are not food in the catalogue (cleaning products, bags, fees, tax, totals, discounts) go in "unmatched" by their printed name, or are skipped if they are fees, tax or totals.
- Never invent lines that aren't there. If you can't read the receipt, reply {"error":"unreadable"}.
Reply with only JSON: {"store": string|null, "total": number|null, "items": [{"id": string, "text": string, "qty": number, "price": number|null}], "unmatched": [string]}`;

const FRIDGE = `You look at a photo of the inside of a fridge (or pantry shelf) and estimate what food is there.
You get the household's product catalogue (id, name, package unit). Only report catalogue items you can actually see.
Rules:
- qty is in CATALOGUE units and can be a fraction (half a 1 L carton of milk -> 0.5; 6 eggs when the unit is 12 count -> 0.5).
- confidence is "high" when the item and amount are clearly visible, "low" when partly hidden or the amount is a guess.
- Don't guess about closed drawers or opaque containers. If the photo is not a fridge or shelf, reply {"error":"unreadable"}.
Reply with only JSON: {"items": [{"id": string, "text": string, "qty": number, "confidence": "high"|"low"}]}`;

function catalogueOf(p) {
  return (Array.isArray(p.catalogue) ? p.catalogue : []).slice(0, 120)
    .map((c) => ({ id: String(c.id || "").slice(0, 40), name: String(c.name || "").slice(0, 60), unit: String(c.unit || "").slice(0, 20) }))
    .filter((c) => c.id);
}
function imageOf(img) {
  const type = String(img?.media_type || ""), data = String(img?.data || "");
  if (!/^image\/(jpeg|png|webp|gif)$/.test(type)) return { err: "Send a JPEG, PNG or WebP photo.", status: 400 };
  if (data.length > 4_500_000) return { err: "That photo is too large. Try a smaller one.", status: 413 };
  return { block: { type: "image", source: { type: "base64", media_type: type, data } } };
}
async function readShop(p, headers) {
  const kind = p.fridge ? "fridge" : p.receipt ? "receipt" : "email";
  const cat = catalogueOf(p);
  if (!cat.length) return json({ error: "Send the product catalogue with the request." }, 400, headers);
  const ids = new Set(cat.map((c) => c.id));
  const content = [];
  if (kind === "email") {
    const text = String(p.email?.text || "").slice(0, 8000);
    if (!text.trim()) return json({ error: "Paste the order email first." }, 400, headers);
    content.push({ type: "text", text: `Catalogue: ${JSON.stringify(cat)}\n\nOrder email:\n${text}` });
  } else {
    const im = imageOf((p.fridge || p.receipt).image);
    if (im.err) return json({ error: im.err }, im.status, headers);
    content.push(im.block, { type: "text", text: `Catalogue: ${JSON.stringify(cat)}\n\nRead this ${kind === "fridge" ? "fridge photo" : "receipt"}.` });
  }
  try {
    const out = await callClaude(kind === "fridge" ? FRIDGE : BOUGHT, content, 1500);
    if (out.error) return json({ error: kind === "fridge" ? "Couldn't make out the shelves in that photo. Try a brighter shot with the door wide open." : "Couldn't read that receipt. Try a flat, well-lit photo of the whole receipt." }, 422, headers);
    const num = (v) => (v == null || v === "" || isNaN(+v) ? null : Math.max(0, Math.min(100, +v)));
    const items = (out.items || []).slice(0, 80).map((x) => ({
      id: ids.has(String(x.id)) ? String(x.id) : null, text: String(x.text || "").slice(0, 80),
      qty: num(x.qty) ?? 1, price: num(x.price), confidence: x.confidence === "low" ? "low" : "high",
    }));
    return json({ store: out.store ? String(out.store).slice(0, 60) : null, total: num(out.total), items,
      unmatched: (out.unmatched || []).slice(0, 30).map((x) => String(x).slice(0, 60)), model: MODEL }, 200, headers);
  } catch (e) {
    return json({ error: e.message || "Couldn't reach the AI service. Try again in a moment." }, 502, headers);
  }
}

const cors = (origin) => {
  const allowed = (process.env.ALLOWED_ORIGINS || "").split(",").map((s) => s.trim()).filter(Boolean);
  const ok = !allowed.length || allowed.includes(origin);
  return {
    "Access-Control-Allow-Origin": ok ? origin || "*" : allowed[0],
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "content-type",
    "Vary": "Origin",
  };
};
const json = (body, status, headers) =>
  new Response(JSON.stringify(body), { status, headers: { ...headers, "content-type": "application/json" } });

// Turn an API error into a plain message (and log the details for the Netlify function log).
async function apiError(r) {
  let msg = "";
  try { msg = (await r.json())?.error?.message || ""; } catch {}
  console.error("Anthropic API error", r.status, msg);
  if (r.status === 401) return "The AI key isn't valid. Check ANTHROPIC_API_KEY in Netlify.";
  if (/credit balance/i.test(msg)) return "The AI account is out of credits. Add credits in the Claude Console.";
  if (/model/i.test(msg)) return `The AI model isn't available (${msg}). Set ANTHROPIC_MODEL in Netlify.`;
  if (r.status === 429 || r.status === 529) return "The AI is busy right now. Try again in a moment.";
  return `The AI service returned ${r.status}${msg ? ": " + msg.slice(0, 160) : ""}.`;
}

async function callClaude(system, content, maxTokens) {
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "x-api-key": process.env.ANTHROPIC_API_KEY, "anthropic-version": VERSION, "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, max_tokens: maxTokens, system, messages: [{ role: "user", content }] }),
  });
  if (!r.ok) throw new Error(await apiError(r));
  const data = await r.json();
  const text = (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("");
  const m = text.match(/\{[\s\S]*\}/);
  if (!m) throw new Error("The AI answer couldn't be read. Try again.");
  return JSON.parse(m[0]);
}

async function readLabel(p, headers) {
  const type = String(p.image.media_type || "");
  const data = String(p.image.data || "");
  if (!/^image\/(jpeg|png|webp|gif)$/.test(type)) return json({ error: "Send a JPEG, PNG or WebP photo." }, 400, headers);
  if (data.length > 4_500_000) return json({ error: "That photo is too large. Try a smaller one." }, 413, headers);
  const hh = { diet: String(p.household?.diet || "").slice(0, 20), avoids: (p.household?.avoids || []).slice(0, 8).map((a) => String(a).slice(0, 30)) };
  try {
    const out = await callClaude(LABEL, [
      { type: "image", source: { type: "base64", media_type: type, data } },
      { type: "text", text: `Household: ${JSON.stringify(hh)}. Read this label.` },
    ], 900);
    if (out.error) return json({ error: "Couldn't read a food label in that photo. Try a closer, well-lit shot of the ingredients or nutrition panel." }, 422, headers);
    out.points = (out.points || []).slice(0, 3).map((x) => String(x).slice(0, 160));
    out.summary = String(out.summary || "").slice(0, 400);
    out.model = MODEL;
    return json(out, 200, headers);
  } catch (e) {
    return json({ error: e.message || "Couldn't reach the AI service. Try again in a moment." }, 502, headers);
  }
}

export default async (req) => {
  const headers = cors(req.headers.get("origin") || "");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers });
  if (req.method !== "POST") return json({ error: "Use POST." }, 405, headers);
  if (!process.env.ANTHROPIC_API_KEY) return json({ error: "The AI isn't set up yet: add ANTHROPIC_API_KEY in Netlify." }, 503, headers);

  let p;
  try { p = await req.json(); } catch { return json({ error: "Send the product as JSON." }, 400, headers); }
  if (p.image) return readLabel(p, headers);
  if (p.receipt || p.fridge || p.email) return readShop(p, headers);
  const clip = (v, n) => String(v ?? "").slice(0, n);
  const product = {
    name: clip(p.name, 120), brand: clip(p.brand, 80), serving: clip(p.serving, 60), pack: clip(p.pack, 60),
    ingredients: clip(p.ingredients, 1500), allergens: (p.allergens || []).slice(0, 12).map((a) => clip(a, 30)),
    nutrition_per_serving: p.per_serving || null, nutrition_per_100g: p.per_100g || null,
    household: { diet: clip(p.household?.diet, 20), avoids: (p.household?.avoids || []).slice(0, 8).map((a) => clip(a, 30)) },
  };

  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": process.env.ANTHROPIC_API_KEY, "anthropic-version": VERSION, "content-type": "application/json" },
      body: JSON.stringify({
        model: MODEL, max_tokens: 300, system: SYSTEM,
        messages: [{ role: "user", content: `Product data:\n${JSON.stringify(product)}` }],
      }),
    });
    if (!r.ok) return json({ error: await apiError(r) }, 502, headers);
    const data = await r.json();
    const text = (data.content || []).filter((b) => b.type === "text").map((b) => b.text).join("");
    const m = text.match(/\{[\s\S]*\}/);
    const out = m ? JSON.parse(m[0]) : null;
    if (!out || typeof out.summary !== "string") return json({ error: "The AI answer couldn't be read. Try again." }, 502, headers);
    return json({ summary: out.summary.slice(0, 400), points: (out.points || []).slice(0, 3).map((x) => String(x).slice(0, 160)), model: MODEL }, 200, headers);
  } catch (e) {
    return json({ error: "Couldn't reach the AI service. Try again in a moment." }, 502, headers);
  }
};
