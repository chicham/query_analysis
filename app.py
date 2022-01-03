"""Demo gradio app for some text/query augmentation."""
from __future__ import annotations

import functools
from collections import defaultdict
from itertools import chain
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Sequence

import attr
import environ
import fasttext  # not working with python3.9
import gradio as gr
from tokenizers.pre_tokenizers import Whitespace
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline
from transformers.pipelines.token_classification import AggregationStrategy


def compose(*functions) -> Callable:
    """
    Compose functions.

        Args:
            functions: functions to compose.
        Returns:
            Composed functions.
    """

    def apply(f, g):
        return lambda x: f(g(x))

    return functools.reduce(apply, functions[::-1], lambda x: x)


def mapped(fn) -> Callable:
    """
    Decorator to apply map/filter to a function
    """

    def inner(func):
        partial_fn = functools.partial(fn, func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return partial_fn(*args, **kwargs)

        return wrapper

    return inner


@attr.frozen
class Prediction:
    """Dataclass to store prediction results."""

    label: str
    score: float


@attr.frozen
class Models:
    identification: Predictor
    translation: Predictor
    classification: Predictor
    ner: Predictor
    recipe: Predictor


@attr.frozen
class Predictor:
    load_fn: Callable
    predict_fn: Callable = attr.field(default=lambda model, query: model(query))
    model: Any = attr.field(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "model", self.load_fn())

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.predict_fn(self.model, *args, **kwds)


@environ.config(prefix="QUERY_INTERPRETATION")
class AppConfig:
    @environ.config
    class Identification:
        """Identification model configuration."""

        model = environ.var(default="./models/lid.176.ftz")
        max_results = environ.var(default=3, converter=int)

    @environ.config
    class Translation:
        """Translation models configuration."""

        model = environ.var(default="t5-small")
        sources = environ.var(default="de,fr")
        target = environ.var(default="en")

    @environ.config
    class Classification:
        """Classification model configuration."""

        model = environ.var(default="typeform/distilbert-base-uncased-mnli")
        max_results = environ.var(default=5, converter=int)

    @environ.config
    class NER:
        general = environ.var(
            default="asahi417/tner-xlm-roberta-large-uncased-wnut2017",
        )

        recipe = environ.var(default="adamlin/recipe-tag-model")

    identification: Identification = environ.group(Identification)
    translation: Translation = environ.group(Translation)
    classification: Classification = environ.group(Classification)
    ner: NER = environ.group(NER)


def predict(
    models: Models,
    query: str,
    categories: Sequence[str],
    supported_languages: tuple[str, ...] = ("fr", "de"),
) -> tuple[
    Mapping[str, float],
    str,
    Mapping[str, float],
    Sequence[tuple[str, str | None]],
    Sequence[tuple[str, str | None]],
]:
    """Predict from a textual query:
    - the language
    - classify as a recipe or not
    - extract the recipe
    """

    def predict_lang(query) -> Mapping[str, float]:
        def predict_fn(query) -> Sequence[Prediction]:
            return tuple(
                Prediction(label=label, score=score)
                for label, score in zip(*models.identification(query, k=176))
            )

        @mapped(map)
        def format_label(prediction: Prediction) -> Prediction:
            return attr.evolve(
                prediction,
                label=prediction.label.replace("__label__", ""),
            )

        def filter_labels(prediction: Prediction) -> bool:
            return prediction.label in supported_languages + ("en",)

        def format_output(predictions: Sequence[Prediction]) -> dict:
            return {pred.label: pred.score for pred in predictions}

        apply_fn = compose(
            predict_fn,
            format_label,
            functools.partial(filter, filter_labels),
            format_output,
        )
        return apply_fn(query)

    def translate_query(query: str, languages: Mapping[str, float]) -> str:
        def predicted_language() -> str:
            return max(languages.items(), key=lambda lang: lang[1])[0]

        def translate(query):
            lang = predicted_language()
            if lang in supported_languages:
                output = models.translation(query, lang)[0]["translation_text"]
            else:
                output = query

            return output

        return translate(query)

    def classify_query(query, categories) -> Mapping[str, float]:
        predictions = models.classification(query, categories)
        return dict(zip(predictions["labels"], predictions["scores"]))

    def extract_entities(
        predict_fn: Callable,
        query: str,
    ) -> Sequence[tuple[str, str | None]]:
        def get_entity(pred: Mapping[str, str]):
            return pred.get("entity", pred.get("entity_group", None))

        mapping = defaultdict(lambda: None)
        mapping.update(**{pred["word"]: get_entity(pred) for pred in predict_fn(query)})

        query_processed = Whitespace().pre_tokenize_str(query)
        res = tuple(
            chain.from_iterable(
                ((word, mapping[word]), (" ", None)) for word, _ in query_processed
            ),
        )
        print(res)
        return res

    languages = predict_lang(query)
    translation = translate_query(query, languages)
    classifications = classify_query(translation, categories)
    general_entities = extract_entities(models.ner, query)
    recipe_entities = extract_entities(models.recipe, translation)
    return languages, translation, classifications, general_entities, recipe_entities


def main():
    cfg: AppConfig = AppConfig.from_environ()

    def load_translation_models(
        sources: Sequence[str],
        target: str,
        models: Sequence[str],
    ) -> Pipeline:
        result = {
            src: pipeline(f"translation_{src}_to_{target}", models)
            for src, models in zip(sources, models)
        }
        return result

    def extract_commas_separated_values(value: str) -> Sequence[str]:
        return tuple(filter(None, value.split(",")))

    models = Models(
        identification=Predictor(
            load_fn=lambda: fasttext.load_model(cfg.identification.model),
            predict_fn=lambda model, query, k: model.predict(query, k=k),
        ),
        translation=Predictor(
            load_fn=functools.partial(
                load_translation_models,
                sources=extract_commas_separated_values(cfg.translation.sources),
                target=cfg.translation.target,
                models=["Helsinki-NLP/opus-mt-de-en", "Helsinki-NLP/opus-mt-fr-en"],
            ),
            predict_fn=lambda models, query, src: models[src](query),
        ),
        classification=Predictor(
            load_fn=lambda: pipeline(
                "zero-shot-classification",
                model=cfg.classification.model,
            ),
            predict_fn=lambda model, query, categories: model(query, categories),
        ),
        ner=Predictor(
            load_fn=lambda: pipeline(
                "ner",
                model=cfg.ner.general,
                aggregation_strategy=AggregationStrategy.MAX,
            ),
        ),
        recipe=Predictor(
            load_fn=lambda: pipeline("ner", model=cfg.ner.recipe),
        ),
    )

    iface = gr.Interface(
        fn=lambda query, categories: predict(
            models,
            query.strip(),
            extract_commas_separated_values(categories),
        ),
        examples=[["gateau au chocolat paris"], ["Newyork LA flight"]],
        inputs=[
            gr.inputs.Textbox(label="Query"),
            gr.inputs.Textbox(
                label="categories (commas separated and in english)",
                default="cooking and recipe,traveling,location,information,buy or sell",
            ),
        ],
        outputs=[
            gr.outputs.Label(
                num_top_classes=cfg.identification.max_results,
                type="auto",
                label="Language identification",
            ),
            gr.outputs.Textbox(
                label="English query",
                type="auto",
            ),
            gr.outputs.Label(
                num_top_classes=cfg.classification.max_results,
                type="auto",
                label="Predicted categories",
            ),
            gr.outputs.HighlightedText(label="NER generic"),
            gr.outputs.HighlightedText(label="NER Recipes"),
        ],
        interpretation="default",
    )

    iface.launch(debug=True)


if __name__ == "__main__":
    main()
