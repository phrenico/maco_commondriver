from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class MethodResult:
    method: str
    result_file: str


@dataclass(frozen=True)
class RunStep:
    label: str
    env_name: str
    module: str
    python_cmd: str = 'python'


@dataclass(frozen=True)
class FamilySpec:
    key: str
    title: str
    combined_csv: str
    methods: tuple[MethodResult, ...]
    run_steps: tuple[RunStep, ...]
    combine_step: RunStep

    @property
    def method_names(self):
        return tuple(method.method for method in self.methods)

    @property
    def result_files(self):
        return tuple(method.result_file for method in self.methods)


FAMILY_SPECS: Final[dict[str, FamilySpec]] = {
    'logmaps': FamilySpec(
        key='logmaps',
        title='Logistic Maps',
        combined_csv='logmaps_res.csv',
        methods=(
            MethodResult('PCA', 'pca_res.csv'),
            MethodResult('kPCA', 'kpca_res.csv'),
            MethodResult('ICA', 'ica_res.csv'),
            MethodResult('CCA', 'cca_res.csv'),
            MethodResult('DCCA', 'dcca_res.csv'),
            MethodResult('ShRec', 'shrec_res.csv'),
            MethodResult('SFA', 'sfa_res.csv'),
            MethodResult('DCA', 'dca_res.csv'),
            MethodResult('random', 'random_res.csv'),
            MethodResult('MaCo', 'maco_res.csv'),
            MethodResult('AniSOM', 'anisom_res.csv'),
        ),
        run_steps=(
            RunStep('ICA', 'maco_rev1', 'scripts.experiments.logmaps.gen_ica_res'),
            RunStep('PCA', 'maco_rev1', 'scripts.experiments.logmaps.gen_pca_res'),
            RunStep('KPCA', 'maco_rev1', 'scripts.experiments.logmaps.gen_kpca_res'),
            RunStep('CCA', 'maco_rev1', 'scripts.experiments.logmaps.gen_cca_res'),
            RunStep('DCA', 'dca', 'scripts.experiments.logmaps.gen_dca_res'),
            RunStep('DCCA', 'dcca_env', 'scripts.experiments.logmaps.gen_dcca_res'),
            RunStep('Sh-Rec', 'shrec', 'scripts.experiments.logmaps.gen_shrec_res'),
            RunStep('Random', 'maco_rev1', 'scripts.experiments.logmaps.gen_random_res'),
            RunStep('SFA', 'sfa', 'scripts.experiments.logmaps.gen_sfa_res'),
            RunStep('MaCo', 'maco_rev1', 'scripts.experiments.logmaps.gen_maco_res'),
            RunStep('AniSOM', 'maco_rev1', 'scripts.experiments.logmaps.gen_anisom_res'),
        ),
        combine_step=RunStep('Combine', 'maco_rev1', 'scripts.experiments.logmaps.Z_combine_final_res'),
    ),
    'tentmaps': FamilySpec(
        key='tentmaps',
        title='Tent Maps',
        combined_csv='tentmaps_res.csv',
        methods=(
            MethodResult('PCA', 'pca_res.csv'),
            MethodResult('kPCA', 'kpca_res.csv'),
            MethodResult('ICA', 'ica_res.csv'),
            MethodResult('CCA', 'cca_res.csv'),
            MethodResult('DCCA', 'dcca_res.csv'),
            MethodResult('ShRec', 'shrec_res.csv'),
            MethodResult('SFA', 'sfa_res.csv'),
            MethodResult('DCA', 'dca_res.csv'),
            MethodResult('random', 'random_res.csv'),
            MethodResult('MaCo', 'maco_res.csv'),
            MethodResult('AniSOM', 'anisom_res.csv'),
        ),
        run_steps=(
            RunStep('ICA', 'maco_rev1', 'scripts.experiments.tentmaps.gen_ica_res'),
            RunStep('PCA', 'maco_rev1', 'scripts.experiments.tentmaps.gen_pca_res'),
            RunStep('KPCA', 'maco_rev1', 'scripts.experiments.tentmaps.gen_kpca_res'),
            RunStep('CCA', 'maco_rev1', 'scripts.experiments.tentmaps.gen_cca_res'),
            RunStep('DCA', 'dca', 'scripts.experiments.tentmaps.gen_dca_res'),
            RunStep('DCCA', 'dcca_env', 'scripts.experiments.tentmaps.gen_dcca_res'),
            RunStep('Sh-Rec', 'shrec', 'scripts.experiments.tentmaps.gen_shrec_res'),
            RunStep('Random', 'maco_rev1', 'scripts.experiments.tentmaps.gen_random_res'),
            RunStep('SFA', 'sfa', 'scripts.experiments.tentmaps.gen_sfa_res'),
            RunStep('MaCo', 'maco_rev1', 'scripts.experiments.tentmaps.gen_maco_res'),
            RunStep('AniSOM', 'maco_rev1', 'scripts.experiments.tentmaps.gen_anisom_res'),
        ),
        combine_step=RunStep('Combine', 'maco_rev1', 'scripts.experiments.tentmaps.Z_combine_final_res'),
    ),
    'lorenz': FamilySpec(
        key='lorenz',
        title='Lorenz Systems',
        combined_csv='lorenz_res.csv',
        methods=(
            MethodResult('PCA', 'pca_res.csv'),
            MethodResult('ICA', 'ica_res.csv'),
            MethodResult('CCA', 'cca_res.csv'),
            MethodResult('DCCA', 'dcca_res.csv'),
            MethodResult('ShRec', 'shrec_res.csv'),
            MethodResult('SFA', 'sfa_res.csv'),
            MethodResult('DCA', 'dca_res.csv'),
            MethodResult('random', 'random_res.csv'),
            MethodResult('MaCo', 'maco_res.csv'),
        ),
        run_steps=(
            RunStep('ICA', 'maco_rev1', 'scripts.experiments.lorenz.gen_ica_res'),
            RunStep('PCA', 'maco_rev1', 'scripts.experiments.lorenz.gen_pca_res'),
            RunStep('CCA', 'maco_rev1', 'scripts.experiments.lorenz.gen_cca_res'),
            RunStep('DCA', 'dca', 'scripts.experiments.lorenz.gen_dca_res'),
            RunStep('DCCA', 'dcca_env', 'scripts.experiments.lorenz.gen_dcca_res'),
            RunStep('Sh-Rec', 'shrec', 'scripts.experiments.lorenz.gen_shrec_res'),
            RunStep('Random', 'maco_rev1', 'scripts.experiments.lorenz.gen_random_res'),
            RunStep('SFA', 'sfa', 'scripts.experiments.lorenz.gen_sfa_res'),
            RunStep('MaCo', 'maco_rev1', 'scripts.experiments.lorenz.gen_maco_res'),
        ),
        combine_step=RunStep('Combine', 'maco_rev1', 'scripts.experiments.lorenz.Z_combine_final_res'),
    ),
}

PLOT_FAMILY_ORDER: Final[tuple[str, ...]] = ('logmaps', 'tentmaps', 'lorenz')


def get_family_spec(family_key):
    return FAMILY_SPECS[family_key]


def get_plot_family_specs():
    return tuple(FAMILY_SPECS[family_key] for family_key in PLOT_FAMILY_ORDER)


def get_execution_steps(family_key, stage='all'):
    family_spec = get_family_spec(family_key)
    if stage == 'methods':
        return family_spec.run_steps
    if stage == 'combine':
        return (family_spec.combine_step,)
    if stage == 'all':
        return family_spec.run_steps + (family_spec.combine_step,)
    raise ValueError(f'Unknown stage: {stage}')