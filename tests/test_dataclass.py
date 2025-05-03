import pytest
from dataclasses import dataclass, field, asdict, astuple, replace
from typing import List
import dataclasses
import asyncio

# 1. Basic dataclass creation and field defaults
@dataclass
class Point:
    x: int
    y: int = 0

def test_basic_dataclass():
    p = Point(5)
    assert p.x == 5
    assert p.y == 0
    p2 = Point(2, 3)
    assert p2.x == 2
    assert p2.y == 3

# 2. Field types and default_factory
@dataclass
class Group:
    items: List[int] = field(default_factory=list)

def test_default_factory():
    g1 = Group()
    g2 = Group()
    g1.items.append(1)
    assert g2.items == []  # Ensure default_factory is not shared

# 3. Immutability (frozen dataclasses)
@dataclass(frozen=True)
class FrozenPoint:
    x: int
    y: int

def test_frozen_dataclass():
    fp = FrozenPoint(1, 2)
    assert fp.x == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        fp.x = 10

# 4. Comparison and ordering
@dataclass(order=True)
class Ordered:
    val: int

def test_ordering():
    o1 = Ordered(1)
    o2 = Ordered(2)
    assert o1 < o2
    assert o2 > o1

# 5. Post-init processing
@dataclass
class WithInit:
    x: int
    y: int
    total: int = field(init=False)
    def __post_init__(self):
        self.total = self.x + self.y

def test_post_init():
    w = WithInit(3, 4)
    assert w.total == 7

# 6. Inheritance and field overriding
@dataclass
class Base:
    a: int

@dataclass
class Child(Base):
    b: int

def test_inheritance():
    c = Child(1, 2)
    assert c.a == 1
    assert c.b == 2

# 7. asdict, astuple, replace
@dataclass
class Demo:
    x: int
    y: int

def test_asdict_astuple_replace():
    d = Demo(5, 6)
    assert asdict(d) == {'x': 5, 'y': 6}
    assert astuple(d) == (5, 6)
    d2 = replace(d, y=10)
    assert d2.x == 5 and d2.y == 10

# 8. Async story formatting test
@dataclasses.dataclass
class Story:
    title: str
    score: int
    author: str

async def main(context_vars):
    step1_stories = context_vars.get('step1_stories', None)
    if step1_stories:
        # Normalize story list
        if isinstance(step1_stories, dict) and 'stories' in step1_stories:
            step2_list = step1_stories['stories']
        elif isinstance(step1_stories, list):
            step2_list = step1_stories
        else:
            return {
                'status': 'completed',
                'result': "Unexpected data format received from Hacker News.",
                'next_step': 'Handle unexpected data format'
            }
        formatted = []
        for story in step2_list:
            try:
                title = story.get('title', 'N/A')
                score = int(story.get('points', 0))
                author = story.get('author', 'N/A')
                formatted.append(Story(title=title, score=score, author=author))
            except (ValueError, TypeError) as e:
                # Skip stories with invalid data types
                continue
        sorted_stories = sorted(formatted, key=lambda x: x.score, reverse=True)
        header = "| Title | Score | Author |\n|---|---|---|"
        rows = "\n".join(f"| {s.title} | {s.score} | {s.author} |" for s in sorted_stories)
        table = f"{header}\n{rows}"
        return {'status': 'completed', 'result': table, 'next_step': 'Display formatted table'}
    return {
        'status': 'completed',
        'result': "Could not retrieve stories from Hacker News.",
        'next_step': 'Handle error retrieving stories'
    }

def test_format_stories_table():
    sample = [
        {'title': 'Story1', 'points': '20', 'author': 'Alice'},
        {'title': 'Story2', 'points': 5, 'author': 'Bob'}
    ]
    result = asyncio.run(main({'step1_stories': sample}))
    expected = (
        "| Title | Score | Author |\n"
        "|---|---|---|\n"
        "| Story1 | 20 | Alice |\n"
        "| Story2 | 5 | Bob |"
    )
    assert result['status'] == 'completed'
    assert result['result'] == expected
