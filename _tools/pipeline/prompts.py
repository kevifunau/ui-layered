# -*- coding: utf-8 -*-
"""All model prompts, verbatim from the article.

Keeping them in one module means the replication can be diffed against the article
word by word, and per-case overrides can be injected from ``data/cases/*.json``
without editing algorithm code.
"""

# ---------------------------------------------------------------- 2.2.1 LLM plan
PLAN_PROMPT = """Analyze a game UI and return a compact executable layer plan.
Two images are provided in this order: (1) the original image, (2) an OCR-cleaned working image.
OCR regions are listed with stable ids below. First classify their rendering mode:
- ordinary readable UI copy (labels, values, hints, button captions) remains editable text;
- a logo, wordmark or decorative display lettering whose irregular shape, multicolor fill, thick
  custom outline, embedded mascot/icon or hand-drawn geometry is part of its identity must remain
  a raster image asset. Put only those OCR ids in raster_text_ids.
Do not classify text as raster merely because it is bold, outlined or placed on a button. A normal
font that can be acceptably recreated as text must stay editable. This is a rendering decision, not
an OCR-confidence decision.

The cleaned image initially has all OCR glyphs removed. For each raster_text_id, also emit exactly
one non-text query (prefer kind=logo, role=foreground) using its location in the original image.
The pipeline will restore that artwork before segmentation. Never emit ordinary text as a query.

Find all visible non-text assets that should be independently editable:
- large panels, boards, frames and dialogs whose exposed material/boundary is visible;
- every repeated card, slot, tab or row surface as a separate geometry hint;
- complete action/navigation buttons as one object, including any integrated icon shape;
- independent icons, item artwork, portraits, silhouettes and framed illustrations;
- badges, ribbons, locks, counters and grouped status markers with their own visible shape.

Do not emit the full-screen background, ordinary text, shadows, gaps, separators, speculative objects,
duplicate composite descriptions, or people/props inside one framed illustration. A parent surface
and its foreground child must be separate only when both are real editable assets. Repeated instances
share one query but each instance needs its own hint.

Every hint is passed directly to segmentation:
- bbox_norm=[x0,y0,x1,y1] tightly encloses exactly one instance, normalized to 0..1;
- positive_points_norm has 1-3 points on solid target pixels, not edges, holes or background;
- negative_points_norm is empty unless a real competing object is near/inside the box; when used,
  put 1-2 points on that competitor, never on an overlaid child of an underlying surface;
- inspect each repeated instance separately. Do not mechanically copy relative point positions.
- z_order is back-to-front. Parent surfaces must have a lower z_order than their children.
- parent_query_id is required for every element visually carried by another exported element.
  Use the nearest direct material parent, not merely the largest outer panel: artwork, silhouette,
  badge or label inside a card belongs to that card; the card belongs to its panel. Leave it null
  only for true root elements. This parent chain is exported and also controls occlusion repair.

For every query choose one element_repair_mode:
- "image" for a textured, patterned, illustrated, translucent or gradient-bearing parent material
  whose child elements must later be removed and reconstructed by the image model;
- "surface" only for a nearly uniform flat-colour material that can be reconstructed from measured
  neighbouring pixels;
- "none" for a topmost foreground object or an object with no meaningful material behind children.
This is a material decision. A panel/card/frame with artwork, glow or visible texture is "image".

Choose one background_repair.mode for the pixels hidden by the extracted UI:
- "scene": the hidden area is a real textured/illustrated scene and should be reconstructed from
  the visible scene boundary. The repair image will contain only a black diagnostic hole plus a
  very narrow softened edge; do not rely on a generated prefill.
- "surface": the hidden area is visually close to a uniform or translucent colour/material. Use
  measured boundary colour/low-frequency continuation only; do not invent scenery or texture.
- "none": do not synthesize hidden pixels; preserve the current cleaned background.
Select from the image evidence, not from the number of detected layers.

Return JSON only, with no prose and no fields outside this schema. The original image is provided
only for the text audit; the text-cleaned image is the source for non-text layer planning.
{
  "scene_summary": "short description",
  "background_repair": {"mode": "scene|surface|none", "reason": "short evidence-based reason"},
  "text_corrections": [{
    "text": "visible text missed by OCR",
    "bbox_norm": [0.1,0.1,0.2,0.2],
    "confidence": 0.8
  }],
  "raster_text_ids": ["text_001"],
  "queries": [{
    "id": "snake_case_id",
    "name": "short user-facing name",
    "kind": "panel|card|button|icon|illustration|badge|decoration|logo",
    "role": "container|surface|button|foreground|decoration",
    "element_repair_mode": "image|surface|none",
    "parent_query_id": null,
    "z_order": 0,
    "geometry_hints": [{
      "bbox_norm": [0.1,0.1,0.2,0.2],
      "positive_points_norm": [[0.15,0.15]],
      "negative_points_norm": []
    }]
  }]
}
"""

# ------------------------------------------------- 2.3.1 element atlas inpainting
ATLAS_PROMPT = (
    "Image inpainting task. The first image is the Atlas to be repaired, and the second image is a binary mask of exactly the same dimensions.\n"
    "All white regions in the mask must be inpainted, while all black regions in the mask must remain unchanged. Black areas in the first image also represent missing content. The repaired content must seamlessly continue the surrounding background material of the same element.\n"
    "The black regions are not black content to preserve, nor are they text to be redrawn.\n"
    "Repair only the black regions in the first image. Do not introduce any text, numbers, letters, icons, buttons, characters, objects, logos, lighting effects, or additional UI elements.\n"
    "Use only the immediately adjacent background of the same element as reference. Fill the missing regions with continuous colors, gradients, translucent materials, and textures. Do not reference other elements.\n"
    "All non-black content, element positions, dimensions, shapes, outer contours, boundaries, gray spacing, and the overall layout must remain completely unchanged.\n"
    "Even if a black region has the shape of text, treat it solely as a gap in the parent layer's background. Do not restore it as text or as a rectangle.\n"
    "Output a complete image with exactly the same dimensions as the input."
)

# --------------------------------------- 2.3.1 for mask-less providers (ISS-035)
# The article's atlas prompt addresses a (image, mask) pair.  Seedream has no mask
# parameter, so the hole is painted black and this single-image rewrite keeps every
# constraint of the original ("black = missing", "no text/icons", "non-black unchanged",
# "same size") while pointing at the region with a 0-999 <bbox> token.
ATLAS_PROMPT_SINGLE = (
    "Image inpainting task. The image contains pure-black regions; they represent "
    "missing content, not black artwork.\n"
    "All black regions must be inpainted, while everything else must remain unchanged. "
    "This includes small silhouette-shaped black regions (animals, leaves, icons, "
    "badges): every one of them is a gap, none may stay black. "
    "The repaired content must seamlessly continue the surrounding background material "
    "of the same element.\n"
    "The black regions are not black content to preserve, nor are they text to be redrawn.\n"
    "Repair only the black regions in the image. Do not introduce any text, numbers, "
    "letters, icons, buttons, characters, objects, logos, lighting effects, or additional "
    "UI elements.\n"
    "Use only the immediately adjacent background of the same element as reference. Fill "
    "the missing regions with continuous colors, gradients, translucent materials, and "
    "textures. Do not reference other elements.\n"
    "All non-black content, element positions, dimensions, shapes, outer contours, "
    "boundaries, gray spacing, and the overall layout must remain completely unchanged.\n"
    "Even if a black region has the shape of text, an animal, a plant, a vessel or any "
    "other figure, treat it solely as a gap in the parent layer's background: fill it "
    "with that parent's own plain material (its paper colour, gradient or texture). Do "
    "not restore the figure inside the black region, not even as a faint outline, "
    "sketch, watermark or colour echo.\n"
    "Output a complete image with exactly the same dimensions as the input."
)

# --------------------------------------------------------- 2.3.3 background repair
BG_PROMPT = (
    "Fill the black missing region in this image so the surrounding scene continues naturally.\n"
    "Keep visible characters and objects, and naturally continue anything that reaches the black region.\n"
    "Do not add UI, text, logos, or unrelated objects. Return one complete image at the same size."
)

# Repo extension: ComfyUI needs a conditioning pair; keep it aligned with the
# article's "Do not add UI, text, logos, or unrelated objects."
NEG_PROMPT = "text, letters, numbers, watermark, logo, extra UI elements, unrelated objects"