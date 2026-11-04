from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal, Any
import jsonschema
Action = Literal['call','answer','clarify']
@dataclass(frozen=True)
class SemanticSpec:
    family: str
    operation: int
    action: Action
    values: dict[str, Any]
    missing: str | None = None
    ambiguous: bool = False
@dataclass(frozen=True)
class ModelInput:
    id: str
    user: str
    tools: list[dict]
@dataclass(frozen=True)
class Case:
    id: str
    split: str
    family: str
    cluster: str
    seed: int
    template: int
    opaque: bool
    spec: SemanticSpec
    user: str
    tools: list[dict]
    expected_name: str | None
    expected_arguments: dict[str, Any]
    def model_input(self) -> ModelInput: return ModelInput(self.id,self.user,self.tools)
    def to_dict(self) -> dict: return asdict(self)
    def validate(self) -> None:
        if self.split not in ('discovery','selection','final','transfer'): raise ValueError('invalid split')
        if self.spec.action not in ('call','answer','clarify'): raise ValueError('invalid action')
        if len({t['function']['name'] for t in self.tools}) != len(self.tools): raise ValueError('duplicate function')
        for t in self.tools: jsonschema.Draft202012Validator.check_schema(t['function']['parameters'])
        if self.spec.action == 'call':
            schema=next(t['function']['parameters'] for t in self.tools if t['function']['name']==self.expected_name)
            jsonschema.validate(self.expected_arguments,schema)

def from_dict(d: dict) -> Case:
    d = dict(d); d['spec']=SemanticSpec(**d['spec']); c=Case(**d); c.validate(); return c
