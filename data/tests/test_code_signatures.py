from j1data.code_signatures import strip_license_text

MIT_HEADER = """#!/bin/bash
# MIT License
#
# Copyright (c) 2019 Someone
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software").
"""

APACHE_HEADER = """/*
 * Copyright 2018 Example Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
"""


def test_license_comment_blocks_are_removed_for_signatures() -> None:
    body = 'echo "deploying"\nterraform apply'
    stripped = strip_license_text(MIT_HEADER + "\n" + body)
    assert "Permission" not in stripped and "terraform apply" in stripped
    java = APACHE_HEADER + "public class A {\n  int x = 1;\n}"
    assert strip_license_text(java).strip().startswith("public class A")


def test_ordinary_comments_and_markdown_license_sections() -> None:
    code = "# compute the mean of a list\ndef mean(xs):\n    return sum(xs) / len(xs)"
    assert strip_license_text(code) == code
    readme = "# Tool\n\nDoes things.\n\n## License\n\nMIT, see LICENSE.\n\n## Usage\n\nRun it."
    stripped = strip_license_text(readme)
    assert "MIT" not in stripped and "Run it." in stripped and "Does things." in stripped


def test_file_that_is_only_a_license_keeps_its_text() -> None:
    assert strip_license_text(MIT_HEADER) == MIT_HEADER
