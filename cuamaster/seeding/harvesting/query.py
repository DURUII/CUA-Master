"""Build search queries from the taxonomy (category -> subcategory -> app)."""
from __future__ import annotations

from dataclasses import dataclass

from ..common.config import SourceConfig


@dataclass
class QuerySpec:
    """A single search query plus its three-layer taxonomy attribution."""

    text: str
    category: str
    subcategory: str
    application: str


def build_queries(
    config: SourceConfig,
    category: str | None = None,
    app: str | None = None,
) -> list[QuerySpec]:
    """Cartesian product of leaf apps x keywords, tagged with their taxonomy.

    When keywords is empty, fall back to the application name alone.
    When category is given, restrict to that single top-level category
    (case-insensitive); raise if it does not exist in the taxonomy.
    When app is given, restrict to that single leaf application
    (case-insensitive), reusing its taxonomy labels; raise if not found.
    """
    taxonomy = config.taxonomy
    if category is not None:
        match = {c.lower(): c for c in taxonomy}.get(category.lower())
        if match is None:
            raise ValueError(
                f"category {category!r} not found; available: {sorted(taxonomy)}"
            )
        taxonomy = {match: taxonomy[match]}

    keywords = config.keywords or [""]
    specs: list[QuerySpec] = []
    for category, subcategories in taxonomy.items():
        for subcategory, apps in subcategories.items():
            for leaf in apps:
                if app is not None and leaf.lower() != app.lower():
                    continue
                for keyword in keywords:
                    specs.append(
                        QuerySpec(
                            text=f"{leaf} {keyword}".strip(),
                            category=category,
                            subcategory=subcategory,
                            application=leaf,
                        )
                    )

    if app is not None and not specs:
        all_apps = sorted(
            a for subs in config.taxonomy.values() for apps in subs.values() for a in apps
        )
        raise ValueError(f"app {app!r} not found in taxonomy; available: {all_apps}")

    return specs
