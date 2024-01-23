# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Naver movie review corpus for binary sentiment classification"""


import csv
import os
import requests
from tqdm import tqdm
from tarfile import TarFile
from zipfile import ZipFile
from pathlib import Path
import datasets
import json
from datasets import Image, Sequence, Value, Features
from pySmartDL import SmartDL


_CITATION = """"""

_DESCRIPTION = """\
인간이 가진 상식적인 지식이나 배경지식을 바탕으로, 이미지에 관련한 질문에 대해 이미지 속에서 답을 찾아야 하는 태스크
"""


_LICENSE = ""

DATASET_KEY = "71357"
BASE_DOWNLOAD_URL = f"https://api.aihub.or.kr/down/{DATASET_KEY}.do"
_HOMEPAGE = f"https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn={DATASET_KEY}"


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class Outside_Knowledge_based_Multimodal_QA_Data(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = Features(
            {
                "IMAGE_ID": Value("string"),
                "IMAGE_NAME": Value("string"),
                "KOREAN_IMAGE": Value("string"),
                "DIRECT_IMAGE": Value("string"),
                "ACTION": Value("string"),
                "CAPTION": Value("string"),
                "IMAGE_ACQUISITION_DATE": Value("string"),
                "IMAGE_URL": Value("string"),
                "LICENSE": Value("string"),
                "MAINOBJECT": Value("string"),
                "MAINOBJECT_URL": Value("string"),
                "SCENE": Value("string"),
                "SUBOBJECT_1": Value("string"),
                "SUBOBJECT_2": Value("string"),
                "WIDTH": Value("string"),
                "HEIGHT": Value("string"),
                "WIKI_FILE": Value("string"),
                "bounding_box": [
                    {
                        "BOX_ID": Value("int32"),
                        "OBJECT": Value("string"),
                        "X_COORDINATE": Value("int32"),
                        "Y_COORDINATE": Value("int32"),
                        "BOX_WIDTH": Value("int32"),
                        "BOX_HEIGHT": Value("int32"),
                    }
                ],
                "questions": [
                    {
                        "question_id": Value("string"),
                        "visual_concept": Sequence(feature=[Value("string")]),
                        "question_ko": Value("string"),
                        "question_en": Value("string"),
                        "answer_ko": Value("string"),
                        "answer_en": Value("string"),
                        "question_type": Sequence(feature=[Value("string")]),
                        "ans_source": Value("string"),
                        "kb_source": Sequence(feature=[Value("string")]),
                        "fact": Sequence(feature=[Value("string")]),
                    }
                ],
                "IMAGE": Image(),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def aihub_downloader(self, recv_path: Path):
        aihub_id = os.getenv("AIHUB_ID")
        aihub_pass = os.getenv("AIHUB_PASS")

        header = {
            "id": aihub_id,
            "pass": aihub_pass,
        }
        param = "fileSn=all"

        obj = SmartDL(
            f"{BASE_DOWNLOAD_URL}?{param}",
            progress_bar=True,
            request_args={"headers": header},
            dest=str(recv_path),
        )
        obj.start(blocking=True)

    def unzip_data(self, tar_file: Path, unzip_dir: Path) -> list:
        with TarFile(tar_file, "r") as mytar:
            mytar.extractall(unzip_dir)
            os.remove(tar_file)

        part_glob = Path(unzip_dir).rglob("*.zip.part*")

        part_dict = dict()
        for part_path in part_glob:
            parh_stem = str(part_path.parent.joinpath(part_path.stem))

            if parh_stem not in part_dict:
                part_dict[parh_stem] = list()

            part_dict[parh_stem].append(part_path)

        for dst_path, part_path_ls in part_dict.items():
            with open(dst_path, "wb") as byte_f:
                for part_path in sorted(part_path_ls):
                    byte_f.write(part_path.read_bytes())
                    os.remove(part_path)

        return list(unzip_dir.rglob("*.zip*"))

    def _split_generators(self, dl_manager):
        data_name = "Outside_Knowledge_based_Multimodal_QA_Data"
        cache_dir = Path(dl_manager.download_config.cache_dir)
        unzip_dir = cache_dir.joinpath(data_name)

        if not unzip_dir.exists():
            tar_file = cache_dir.joinpath(f"{data_name}.tar")
            self.aihub_downloader(tar_file)
            zip_file_path = self.unzip_data(tar_file, unzip_dir)

        train_split = [x for x in zip_file_path if "Training" in str(x)]
        valid_split = [x for x in zip_file_path if "Validation" in str(x)]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train_split,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": valid_split,
                    "split": "valid",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        source_ls = [ZipFile(x) for x in filepath if "원천데이터" in str(x)]
        label_ls = [ZipFile(x) for x in filepath if "라벨링데이터" in str(x) and "VQA" in str(x)]

        label = json.loads(label_ls[0].open(label_ls[0].filelist[0]).read().decode("utf-8"))
        label = {x["IMAGE_NAME"]: x for x in label["annotations"]}
        for zip_file in source_ls:
            for file in zip_file.filelist:
                filename = file.filename.replace("/", "")
                label[filename]["IMAGE"] = zip_file.open(file.filename).read()
                image_id = label[filename]["IMAGE_ID"]

                yield (int(image_id), label[filename])
