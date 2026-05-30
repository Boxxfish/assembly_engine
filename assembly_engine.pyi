from typing_extensions import Self

class PyVec3:
    x: float
    y: float
    z: float

    def __new__(
        cls,
        x: float,
        y: float,
        z: float,
    ) -> PyVec3: ...

class PyQuat:
    x: float
    y: float
    z: float
    w: float

    def __new__(
        cls,
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> PyQuat: ...

class Part:
    @staticmethod
    def from_json(json_str: str) -> "Part": ...
    def to_json(self) -> str: ...

class PyPlacement:
    part_index: int
    position: PyVec3
    rotation: PyQuat

    def __new__(
        cls, part_index: int, position: PyVec3, rotation: PyQuat
    ) -> PyPlacement: ...

class Query:
    part_id: int | None
    anchor_idx: int | None
    single: bool

    def __new__(
        cls, part_id: int | None, anchor_idx: int | None, single: bool
    ) -> Self: ...

class PyAssembledModel:
    placements: list[PyPlacement]

class AssemblyEngineConfig:
    def __new__(
        cls, num_candidate_turns: int, bar_increment_every: float
    ) -> AssemblyEngineConfig: ...

class EngineState:
    pass

class PyAssemblyEngine:
    def __new__(
        cls, parts: list[Part], config: AssemblyEngineConfig
    ) -> PyAssemblyEngine: ...
    def clear(self) -> None: ...
    def get_model(self) -> PyAssembledModel: ...
    def load_model(self, model: PyAssembledModel) -> None: ...
    def get_parts(self) -> list[Part]: ...
    def add_placement(self, placement: PyPlacement): ...
    def query(self, query: Query) -> list[PyPlacement]: ...
    def query_multi(self, queries: list[Query]) -> list[list[PyPlacement]]: ...
    def query_exists(self, part_id: int | None, anchor_idx: int | None) -> bool: ...
    def query_exists_multi(self, queries: list[Query]) -> list[bool]: ...
    def get_state(self) -> EngineState: ...
    def load_state(self, state: EngineState) -> None: ...