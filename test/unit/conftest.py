# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_path() -> Path:
    return Path("test/data")
