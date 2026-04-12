# IG Collection to Obsidian

Small Python CLI for turning an Instaloader `:saved` archive into Obsidian notes with linked media, saved-collection tags, optional Faster-Whisper transcripts, and optional Ollama categorization.

## What It Does

- Runs `instaloader` against your saved Instagram posts if you want the project to own the download step.
- Scans an existing Instaloader archive and groups files into posts by shortcode.
- Reads adjacent caption text, transcript files, and JSON metadata when they exist.
- Applies collection labels from a local `collections.json`.
- Extracts locations from Instagram metadata or a manual `locations.json`.
- Classifies posts into your own taxonomy using caption/transcript text when AI categorization is enabled.
- Symlinks or copies media into an Obsidian-friendly folder.
- Writes one markdown note per post/reel with YAML frontmatter and embedded media links.
- Exports a Google My Maps compatible CSV with names, descriptions, and coordinates/addresses.

## Project Layout

```text
archive/
  saved/
    someuser/
      2026-04-08_12-01-02_ABC123.mp4
      2026-04-08_12-01-02_ABC123.txt
      2026-04-08_12-01-02_ABC123.json.xz
      2026-04-08_12-01-02_ABC123.transcript.txt

vault/
  instagram/
    notes/
      2026-04-08_ABC123.md
    media/
      someuser/
        2026-04-08_12-01-02_ABC123.mp4
    exports/
      google-my-maps.csv
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[transcribe]'
cp config.example.json config.json
cp collections.example.json collections.json
cp locations.example.json locations.json
cp taxonomy.example.json taxonomy.json
```

If you do not need editable install mode, this also works:

```bash
pip install -r requirements.txt
```

Edit `config.json`:

- `paths.archive_dir`: your Instaloader output folder, usually `~/archive/instagram/saved`
- `paths.vault_dir`: the Obsidian folder you want this tool to manage
- `download.instagram_username`: your Instagram username if you want `ig-obsidian download` / `sync --download` to run Instaloader for you
- `paths.locations_file`: optional per-shortcode location overrides and descriptions for Google Maps export
- `categorization.taxonomy_file`: your category taxonomy for AI classification
- `categorization.model`: Ollama model name, defaults to `gemma4:e4b`
- `categorization.base_url`: Ollama server URL, overridden by `OLLAMA_HOST` when set
- `categorization.dynamic_location_categories`: generic categories that should expand to `<category>/<location>` when the post is clearly about a place

## Collection Mapping

`collections.json` supports either shape:

```json
{
  "ABC123": "travels",
  "XYZ789": "food"
}
```

or:

```json
{
  "travels": ["ABC123", "https://www.instagram.com/reel/XYZ789/"],
  "food": ["LMN456"]
}
```

If you do not have an automated way to export Instagram collections yet, keep this file manual. That was the weak point in the original plan too.

## Location Mapping

`locations.json` is optional, but it is the cleanest way to add descriptions and fix missing coordinates. It is keyed by shortcode and can contain one location object or a list:

```json
{
  "ABC123": {
    "name": "Cafe Example",
    "description": "Good brunch spot from a saved reel.",
    "address": "123 Example Street, Lisbon, Portugal",
    "latitude": 38.7223,
    "longitude": -9.1393
  }
}
```

If Instagram metadata already contains a location, the tool will pick it up automatically. Manual overrides win when both exist.

## AI Taxonomy

`taxonomy.json` defines the generic categories the model is allowed to use. It can be a list of names or an object of `name -> description`. Descriptions produce better results.

```json
{
  "travel": "Destinations, itineraries, landmarks, hotels, flights, or general trip inspiration.",
  "food": "Restaurants, cafes, recipes, dishes, bakeries, bars, or local food recommendations.",
  "shopping": "Markets, stores, boutiques, malls, and products worth buying."
}
```

By default, `travel` and `food` are also treated as location-aware roots. When a saved post is clearly about a place, the model can return categories like `travel/Kyoto, Japan` or `food/Lisbon, Portugal`. Other content still stays in the generic taxonomy. If you want a strict fixed taxonomy again, set `categorization.dynamic_location_categories` to `[]`.

When categorization is enabled, the tool:

- reads caption/transcript/location context
- classifies into your taxonomy with Ollama tool calling
- uses dynamic `category/location` values for configured location-aware categories when a place is clear
- writes a `.ai_categories.json` sidecar next to the source files
- writes a `.ai_categories.error.txt` marker when Ollama returns malformed per-post output
- skips already-classified posts on future reruns unless `overwrite` is true
- skips previous categorization failure markers on future reruns unless `overwrite` is true
- writes the chosen categories into note frontmatter and an `## AI Categories` section

Categorization uses the local Ollama `/api/chat` endpoint with `think=false` to avoid extra reasoning latency during classification. `categorization.model` defaults to `gemma4:e4b`, and `OLLAMA_MODEL` / `OLLAMA_HOST` override the config at runtime.

## Usage

Run the full pipeline:

```bash
ig-obsidian sync --config config.json
```

Force a fresh download first:

```bash
ig-obsidian sync --config config.json --download
```

Only transcribe videos:

```bash
ig-obsidian transcribe --config config.json
```

Only rebuild notes and media links:

```bash
ig-obsidian build --config config.json
```

Only run AI categorization:

```bash
ig-obsidian categorize --config config.json
```

Export locations for Google My Maps:

```bash
ig-obsidian export-maps --config config.json
```

Only run Instaloader:

```bash
ig-obsidian download --config config.json
```

## Notes

- `download` shells out to the `instaloader` CLI. The package is installed as a dependency, but the login flow is still interactive.
- Transcription only supports `faster-whisper` right now. That keeps the implementation simpler and avoids the extra `ffmpeg` requirement from the reference Whisper package.
- The first transcription run may pause on model initialization while `faster-whisper` downloads model files from Hugging Face. The CLI now prints that step explicitly.
- AI categorization uses Ollama's local chat API with a `categorize_saved_post` function schema. Make sure your Ollama server is running and the model is available, for example:

```bash
ollama pull gemma4:e4b
export OLLAMA_MODEL=gemma4:e4b
ig-obsidian categorize --config config.json
```

- Ollama powers taxonomy classification only. Video transcription still uses `faster-whisper`.
- Media is mirrored into the Obsidian folder using symlinks by default. Switch `obsidian.use_symlinks` to `false` if you want copies instead.
- Notes are regenerated deterministically. Re-running `build` or `sync` updates the markdown in place.
- Posts are deduped by Instagram shortcode, and per-post media variants are deduped by their sidecar slot so reruns do not create extra notes for the same reel.
- AI category sidecars are persisted next to the source archive so future reruns skip already-classified posts unless you force overwrite.
- Google Maps import is meant for [Google My Maps](https://support.google.com/mymaps/answer/3024836), which accepts CSV imports. After importing there, you can still view the map in Google Maps.
- On macOS, use `caffeinate` for long runs if you plan to leave the machine unattended:

```bash
caffeinate -dimsu ig-obsidian sync --config config.json
```

## Tests

```bash
python3 -m unittest discover -s tests
```
