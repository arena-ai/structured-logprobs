from typing import Any, TypeAlias

from lark import Lark, Token, Transformer_NonRecursive, Tree, v_args
from lark.tree import Meta
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from pydantic import BaseModel

PyTree: TypeAlias = Any  # a tree-like structure built out of container-like Python objects.


class HasProb(BaseModel):
    value: Any
    start: int
    end: int
    logprob: float


# Define a grammar for JSON
json_grammar = r"""
    start: value

    ?value: object              #'?' is a Lark convention indicating that the rule can return the value directly instead of creating a separate parse tree node.
          | array
          | string
          | SIGNED_NUMBER -> number    #'-> number' specifies an alias for the rule
          | "true"
          | "false"
          | "null"

    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : key ":" value
    key    : ESCAPED_STRING

    string : ESCAPED_STRING

    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""


# Transformer that processes the tree and substitutes each atomic value with the cumulative log-probability of its tokens
@v_args(meta=True)
class Extractor(Transformer_NonRecursive):
    def __init__(self, tokens: list[ChatCompletionTokenLogprob], token_indices: list[int]):
        super().__init__()
        self.tokens = tokens
        self.token_indices = token_indices

    def _compute_logprob_sum(self, start: int, end: int) -> float:
        token_start = self.token_indices[start]
        token_end = self.token_indices[end]
        sum_logporb = sum(self.tokens[i].logprob for i in range(token_start, token_end))
        return sum_logporb

    def number(self, meta: Meta, children: list[Token]) -> float:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return logprob_sum

    def string(self, meta: Meta, children: list[Token]) -> float:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return logprob_sum

    def true(self, meta: Meta, children: list[Token]) -> float:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return logprob_sum

    def false(self, meta: Meta, children: list[Token]) -> float:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return logprob_sum

    def null(self, meta: Meta, children: list[Token]) -> None:
        return None

    def array(self, meta: Meta, children: list[Any]) -> list[float]:
        return children

    def object(self, meta: Meta, children: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in children:
            result[key] = value
        return result

    def pair(self, meta: Meta, children: list[Any]) -> tuple[str, Any]:
        value = children[1]
        key = children[0]
        if isinstance(value, Tree) and not value.children:  # ['b', Tree(Token('RULE', 'value'), [])]
            value = None
        return key, value

    def key(self, meta: Meta, children: list[Token]) -> str:
        return children[0][1:-1]

    def start(self, meta: Meta, children: list[dict[str, Any]]) -> dict[str, Any]:
        return children[0]


def extract_json_data(json_string: str, tokens: list[ChatCompletionTokenLogprob], token_indices: list[int]) -> PyTree:
    json_parser = Lark(json_grammar, parser="lalr", propagate_positions=True, maybe_placeholders=False)
    tree = json_parser.parse(json_string)
    extractor = Extractor(tokens, token_indices)
    return extractor.transform(tree)


# Transformer that embeds log-probabilities for atomic values as in-line fields in dictionaries
@v_args(meta=True)
class ExtractorInline(Transformer_NonRecursive):
    def __init__(self, tokens: list[ChatCompletionTokenLogprob], token_indices: list[int]):
        super().__init__()
        self.tokens = tokens
        self.token_indices = token_indices

    def _compute_logprob_sum(self, start: int, end: int) -> float:
        token_start = self.token_indices[start]
        token_end = self.token_indices[end]
        sum_logporb = sum(self.tokens[i].logprob for i in range(token_start, token_end))
        return sum_logporb

    def number(self, meta: Meta, children: list[Token]) -> HasProb:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return HasProb(value=float(children[0]), start=meta.start_pos, end=meta.end_pos, logprob=logprob_sum)

    def string(self, meta: Meta, children: list[Token]) -> HasProb:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return HasProb(value=children[0][1:-1], start=meta.start_pos, end=meta.end_pos, logprob=logprob_sum)

    def true(self, meta: Meta, children: list[Token]) -> HasProb:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return HasProb(value=True, start=meta.start_pos, end=meta.end_pos, logprob=logprob_sum)

    def false(self, meta: Meta, children: list[Token]) -> HasProb:
        logprob_sum = self._compute_logprob_sum(meta.start_pos, meta.end_pos)
        return HasProb(value=False, start=meta.start_pos, end=meta.end_pos, logprob=logprob_sum)

    def null(self, meta: Meta, children: list[Token]) -> None:
        return None

    def array(self, meta: Meta, children: list[dict[str, Any] | Any]) -> list[dict[str, Any] | Any]:
        return [child.value if isinstance(child, HasProb) else child for child in children]

    def object(self, meta: Meta, children: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in children:
            if isinstance(value, HasProb):
                result[key] = value.value
                result[f"{key}_logprob"] = value.logprob
            else:
                result[key] = value
        return result

    def pair(self, meta: Meta, children: list[str | Any]) -> tuple[str, Any]:
        value = children[1]
        key = children[0]
        if isinstance(value, Tree) and not value.children:  # ['b', Tree(Token('RULE', 'value'), [])]
            value = None
        return key, value

    def key(self, meta: Meta, children: list[Token]) -> str:
        return children[0][1:-1]

    def start(self, meta: Meta, children: list[dict[str, Any]]) -> dict[str, Any]:
        return children[0]


def extract_json_data_inline(
    json_string: str, tokens: list[ChatCompletionTokenLogprob], token_indices: list[int]
) -> PyTree:
    json_parser = Lark(json_grammar, parser="lalr", propagate_positions=True, maybe_placeholders=False)
    tree = json_parser.parse(json_string)
    extractor = ExtractorInline(tokens, token_indices)
    return extractor.transform(tree)
