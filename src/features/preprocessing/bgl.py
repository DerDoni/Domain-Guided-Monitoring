import dataclass_cli
import dataclasses
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Set
import re
from .base import Preprocessor
from collections import Counter
from .drain import Drain, DrainParameters
import numpy as np
from src.features.preprocessing.spell import Spell, SpellParameters
from src.features.preprocessing.nulog import Nulog, NulogParameters


@dataclass_cli.add
@dataclasses.dataclass
class BGLPreprocessorConfig:
    aggregated_log_file: Path = Path("data/logs_BGL.csv")
    final_log_file: Path = Path("data/bgl.pkl")
    relevant_aggregated_log_columns: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Level",
            "Code1",
            "Code2",
            "Component1",
            "Component2",
            "Label",
        ],
    )
    aggregate_per_max_number: int = -1
    aggregate_per_time_frequency: str = ""
    log_datetime_column_name: str = "timestamp"
    log_datetime_format: str = "%Y-%m-%d %H:%M:%S.%f"
    log_payload_column_name: str = "Payload"
    label_column_name: str = "Label"
    use_log_hierarchy: bool = False
    fine_drain_log_depth: int = 10
    fine_drain_log_st: float = 0.75
    medium_drain_log_depth: int = 7
    medium_drain_log_st: float = 0.4
    coarse_drain_log_depth: int = 4
    coarse_drain_log_st: float = 0.2
    drain_log_depths: List[int] = dataclasses.field(default_factory=lambda: [],)
    drain_log_sts: List[float] = dataclasses.field(default_factory=lambda: [],)
    add_log_clusters: bool = True
    relevant_log_column: str = "fine_log_cluster_template"
    log_template_file: Path = Path("data/attention_log_templates.csv")
    log_parser: str = "drain"
    parser_combination: str = ""


class BGLLogsPreprocessor(Preprocessor):
    sequence_column_name: str = "all_events"

    def __init__(self, config: BGLPreprocessorConfig):
        self.config = config
        self.relevant_columns = set(
            [x for x in self.config.relevant_aggregated_log_columns]
        )
        if self.config.log_parser != "all":
            self.relevant_columns.add("fine_log_cluster_template")
            self.relevant_columns.add("medium_log_cluster_template")
            self.relevant_columns.add("coarse_log_cluster_template")
        else:
            self.relevant_columns.add("fine_log_cluster_template_drain")
            self.relevant_columns.add("coarse_log_cluster_template_drain")
            self.relevant_columns.add("medium_log_cluster_template_drain")
            self.relevant_columns.add("fine_log_cluster_template_spell")
            self.relevant_columns.add("coarse_log_cluster_template_spell")
            self.relevant_columns.add("medium_log_cluster_template_spell")
            self.relevant_columns.add("fine_log_cluster_template_nulog")
            self.relevant_columns.add("coarse_log_cluster_template_nulog")
            self.relevant_columns.add("medium_log_cluster_template_nulog")

        for i in range(len(self.config.drain_log_depths)):
            self.relevant_columns.add(str(i) + "_log_cluster_template")

    def load_data(self, max_data_size=-1) -> pd.DataFrame:
        log_only_data = self._load_log_only_data(max_data_size)
        log_only_data["grouper"] = 1
        return self._aggregate_per(log_only_data, aggregation_column="grouper")

    def _load_log_only_data(self, max_data_size=-1) -> pd.DataFrame:
        log_df = self._read_log_df(max_data_size=max_data_size)
        for column in [x for x in log_df.columns if "log_cluster_template" in x]:
            log_df[column] = (
                log_df[column]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
                .apply(lambda x: x if len(x) > 0 else "___empty___")
            )
        return log_df

    def _aggregate_per(
        self, merged_df: pd.DataFrame, aggregation_column: str = "parent_trace_id"
    ) -> pd.DataFrame:
        logging.debug("Aggregating BGL data per %s", aggregation_column)
        for column in self.relevant_columns:
            merged_df[column] = merged_df[column].apply(
                lambda x: column + "#" + x.lower() if len(x) > 0 else ""
            )

        merged_df["all_events"] = merged_df[list(self.relevant_columns)].values.tolist()
        merged_df["templates"] = merged_df[[x for x in self.relevant_columns if "log_cluster_template" in x]].values.tolist()
        merged_df["attributes"] = merged_df[
            [x for x in self.relevant_columns if not "log_cluster_template" in x]
        ].values.tolist()
        for log_template_column in [
            x for x in self.relevant_columns if "log_cluster_template" in x
        ]:
            merged_df[log_template_column] = merged_df[log_template_column].apply(
                lambda x: [x]
            )
        events_per_trace = (
            merged_df.sort_values(by="timestamp")
            .groupby(aggregation_column)
            .agg(
                {
                    column_name: lambda x: list(x)
                    for column_name in ["all_events", "attributes", "templates", "label"]
                    + [x for x in self.relevant_columns if "log_cluster_template" in x]
                }
            )
            .reset_index()
        )
        events_per_trace["num_logs"] = events_per_trace[
            self.config.relevant_log_column
        ].apply(lambda x: len([loglist for loglist in x if len(loglist[0]) > 0]))
        events_per_trace["num_events"] = events_per_trace[
            self.config.relevant_log_column
        ].apply(lambda x: len(x))
        return events_per_trace[
            ["num_logs", "num_events", "all_events", "attributes", "templates", "label"]
            + [x for x in self.relevant_columns if "log_cluster_template" in x]
        ]

    def _read_log_df(self, max_data_size=-1) -> pd.DataFrame:
        df = (
            pd.read_csv(self.config.aggregated_log_file)
            .fillna("")
            .astype(str)
            .replace(np.nan, "", regex=True)
        )

        if max_data_size > 0 and max_data_size < df.shape[0]:
            logging.info(
                "Only using first %d rows of log_df with %d rows",
                max_data_size,
                df.shape[0],
            )
            df = df.head(max_data_size)

        rel_df = df[
            self.config.relevant_aggregated_log_columns
            + [self.config.log_datetime_column_name]
            + [self.config.log_payload_column_name]
        ]

        if self.config.log_parser in ["drain", "all"]:
            rel_df = self._add_log_drain_clusters(rel_df)
        if self.config.log_parser in ["spell", "all"]:
            rel_df = self._add_log_spell_clusters(rel_df)
        if self.config.log_parser in ["nulog", "all"]:
            rel_df = self._add_log_nulog_clusters(rel_df)

        if self.config.log_template_file.exists():
            rel_df = self._add_precalculated_log_templates(rel_df)
        rel_df["timestamp"] = pd.to_datetime(
            rel_df[self.config.log_datetime_column_name],
            format=self.config.log_datetime_format
        )
        return rel_df

    def _add_log_drain_clusters_prefix(
        self, log_df: pd.DataFrame, depth: int, st: float, prefix: str
    ) -> pd.DataFrame:
        all_logs_df = pd.DataFrame(
            log_df[self.config.log_payload_column_name].dropna().drop_duplicates()
        )
        drain = Drain(
            DrainParameters(
                depth=depth,
                st=st,
                rex=[
                    ("(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)", ""),
                    ("[^a-zA-Z\d\s:]", ""),
                ],
            ),
            data_df=all_logs_df,
            data_df_column_name=self.config.log_payload_column_name,
        )
        drain_result_df = drain.load_data().drop_duplicates().set_index("log_idx")

        if self.config.log_parser == "drain":
            log_result_df = (
                pd.merge(
                    log_df,
                    pd.merge(
                        all_logs_df,
                        drain_result_df,
                        left_index=True,
                        right_index=True,
                        how="left",
                    )
                    .drop_duplicates()
                    .reset_index(drop=True),
                    on=self.config.log_payload_column_name,
                    how="left",
                )
                .rename(
                    columns={
                        "cluster_template": prefix + "log_cluster_template",
                        "cluster_path": prefix + "log_cluster_path",
                    }
                )
                .drop(columns=["cluster_id"])
            )
            log_result_df[prefix + "log_cluster_template"] = (
                log_result_df[prefix + "log_cluster_template"]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
            )

        else:
            log_result_df = (
                pd.merge(
                    log_df,
                    pd.merge(
                        all_logs_df,
                        drain_result_df,
                        left_index=True,
                        right_index=True,
                        how="left",
                    )
                    .drop_duplicates()
                    .reset_index(drop=True),
                    on=self.config.log_payload_column_name,
                    how="left",
                )
                .rename(
                    columns={
                        "cluster_template": prefix + "log_cluster_template_drain",
                        "cluster_path": prefix + "log_cluster_path",
                    }
                )
                .drop(columns=["cluster_id"])
            )
            log_result_df[prefix + "log_cluster_template_drain"] = (
                log_result_df[prefix + "log_cluster_template_drain"]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
            )
        return log_result_df

    def _add_precalculated_log_templates(self, log_df: pd.DataFrame) -> pd.DataFrame:
        precalculated_templates_df = pd.read_csv(self.config.log_template_file)
        if not "Payload" in precalculated_templates_df.columns:
            logging.error("Invalid log template file - does not contain Payload column!")
            return log_df
        self.relevant_columns.update(
            [x for x in precalculated_templates_df.columns if "log_cluster_template" in x]
        )
        return pd.merge(log_df, precalculated_templates_df, on="Payload", how="left")


    def _add_log_drain_clusters(self, log_df: pd.DataFrame) -> pd.DataFrame:
        log_result_df = self._add_log_drain_clusters_prefix(
            log_df=log_df,
            depth=self.config.fine_drain_log_depth,
            st=self.config.fine_drain_log_st,
            prefix="fine_",
        )
        log_result_df = self._add_log_drain_clusters_prefix(
            log_df=log_result_df,
            depth=self.config.coarse_drain_log_depth,
            st=self.config.coarse_drain_log_st,
            prefix="coarse_",
        )
        log_result_df = self._add_log_drain_clusters_prefix(
            log_df=log_result_df,
            depth=self.config.medium_drain_log_depth,
            st=self.config.medium_drain_log_st,
            prefix="medium_",
        )
        for i in range(len(self.config.drain_log_depths)):
            log_result_df = self._add_log_drain_clusters_prefix(
                log_df=log_result_df,
                depth=self.config.drain_log_depths[i],
                st=self.config.drain_log_sts[i],
                prefix=str(i) + "_",
            )
        return log_result_df
    def _add_log_spell_clusters(self, log_df: pd.DataFrame) -> pd.DataFrame:
        log_result_df = self._add_log_spell_clusters_prefix(log_df=log_df, tau=0.9, prefix="fine_")
        log_result_df = self._add_log_spell_clusters_prefix(log_df=log_result_df, tau=0.7, prefix="medium_")
        log_result_df = self._add_log_spell_clusters_prefix(log_df=log_result_df, tau=0.5, prefix="coarse_")
        return log_result_df
    def _add_log_spell_clusters_prefix(self, log_df: pd.DataFrame, tau: float, prefix: str):
        spell = Spell(
            SpellParameters(tau=tau), data_df=log_df, data_df_column_name=self.config.log_payload_column_name
        )
        spell_result_df = spell.load_data().drop_duplicates().set_index("LineId")

        if self.config.log_parser == "spell":
            spell_result_df = spell_result_df.rename(
                    columns={
                        "EventTemplate": prefix + "log_cluster_template",
                    }
                )
            spell_result_df = spell_result_df.drop(columns=["EventId"])
            spell_result_df[prefix + "log_cluster_template"] = (
                spell_result_df[prefix + "log_cluster_template"]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
            )





        else:
            spell_result_df = spell_result_df.rename(
                    columns={
                        "EventTemplate": prefix + "log_cluster_template_spell",
                    }
                )
            spell_result_df = spell_result_df.drop(columns=["EventId"])
            spell_result_df[prefix + "log_cluster_template_spell"] = (
                spell_result_df[prefix + "log_cluster_template_spell"]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
            )

        return spell_result_df


    def _add_log_nulog_clusters_prefix(self, log_df: pd.DataFrame, k: int, nr_epochs: int, num_samples: int, prefix: str):
        nulog = Nulog(
            NulogParameters(k=k, nr_epochs=nr_epochs, num_samples=num_samples),
            data_df=log_df,
            data_df_column_name=self.config.log_payload_column_name
        )
        nulog_result_df = nulog.load_data().drop_duplicates().set_index("LineId")

        if self.config.log_parser == "nulog":
            nulog_result_df = nulog_result_df.rename(
                    columns={
                        "EventTemplate": prefix + "log_cluster_template",
                    }
                )
            nulog_result_df = nulog_result_df.drop(columns=["EventId"])
            nulog_result_df[prefix + "log_cluster_template"] = (
                nulog_result_df[prefix + "log_cluster_template"]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
            )

        else:
            nulog_result_df = nulog_result_df.rename(
                    columns={
                        "EventTemplate": prefix + "log_cluster_template_nulog",
                    }
                )
            nulog_result_df = nulog_result_df.drop(columns=["EventId"])
            nulog_result_df[prefix + "log_cluster_template_nulog"] = (
                nulog_result_df[prefix + "log_cluster_template_nulog"]
                .fillna("")
                .astype(str)
                .replace(np.nan, "", regex=True)
            )

        return nulog_result_df

    def _add_log_nulog_clusters(self, log_df: pd.DataFrame) -> pd.DataFrame:
        log_result_df = self._add_log_nulog_clusters_prefix(log_df=log_df, k=5, nr_epochs=5 , num_samples=0,prefix="fine_")
        log_result_df = self._add_log_nulog_clusters_prefix(log_df=log_result_df, k=20, nr_epochs=5,num_samples=0, prefix="medium_")
        log_result_df = self._add_log_nulog_clusters_prefix(log_df=log_result_df, k=50, nr_epochs=5, num_samples=0, prefix="coarse_")
        return log_result_df



class AlgorithmChoiceError(Exception):
    """Exception raised for errors when algorithm doesn't exist."""

    def __init__(self, message):
        self.message = message
