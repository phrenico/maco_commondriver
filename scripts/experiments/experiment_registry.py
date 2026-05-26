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
            RunStep('ICA', 'maco_env', 'scripts.experiments.logmaps.gen_ica_res'),
            RunStep('PCA', 'maco_env', 'scripts.experiments.logmaps.gen_pca_res'),
            RunStep('KPCA', 'maco_env', 'scripts.experiments.logmaps.gen_kpca_res'),
            RunStep('CCA', 'maco_env', 'scripts.experiments.logmaps.gen_cca_res'),
            RunStep('DCA', 'dca_env', 'scripts.experiments.logmaps.gen_dca_res'),
            RunStep('DCCA', 'dcca_env', 'scripts.experiments.logmaps.gen_dcca_res'),
            RunStep('Sh-Rec', 'shrec_env', 'scripts.experiments.logmaps.gen_shrec_res'),
            RunStep('Random', 'maco_env', 'scripts.experiments.logmaps.gen_random_res'),
            RunStep('SFA', 'sfa_env', 'scripts.experiments.logmaps.gen_sfa_res'),
            RunStep('MaCo', 'maco_env', 'scripts.experiments.logmaps.gen_maco_res'),
            RunStep('AniSOM', 'maco_env', 'scripts.experiments.logmaps.gen_anisom_res'),
        ),
        combine_step=RunStep('Combine', 'maco_env', 'scripts.experiments.logmaps.Z_combine_final_res'),
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
            RunStep('ICA', 'maco_env', 'scripts.experiments.tentmaps.gen_ica_res'),
            RunStep('PCA', 'maco_env', 'scripts.experiments.tentmaps.gen_pca_res'),
            RunStep('KPCA', 'maco_env', 'scripts.experiments.tentmaps.gen_kpca_res'),
            RunStep('CCA', 'maco_env', 'scripts.experiments.tentmaps.gen_cca_res'),
            RunStep('DCA', 'dca_env', 'scripts.experiments.tentmaps.gen_dca_res'),
            RunStep('DCCA', 'dcca_env', 'scripts.experiments.tentmaps.gen_dcca_res'),
            RunStep('Sh-Rec', 'shrec_env', 'scripts.experiments.tentmaps.gen_shrec_res'),
            RunStep('Random', 'maco_env', 'scripts.experiments.tentmaps.gen_random_res'),
            RunStep('SFA', 'sfa_env', 'scripts.experiments.tentmaps.gen_sfa_res'),
            RunStep('MaCo', 'maco_env', 'scripts.experiments.tentmaps.gen_maco_res'),
            RunStep('AniSOM', 'maco_env', 'scripts.experiments.tentmaps.gen_anisom_res'),
        ),
        combine_step=RunStep('Combine', 'maco_env', 'scripts.experiments.tentmaps.Z_combine_final_res'),
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
            RunStep('ICA', 'maco_env', 'scripts.experiments.lorenz.gen_ica_res'),
            RunStep('PCA', 'maco_env', 'scripts.experiments.lorenz.gen_pca_res'),
            RunStep('CCA', 'maco_env', 'scripts.experiments.lorenz.gen_cca_res'),
            RunStep('DCA', 'dca_env', 'scripts.experiments.lorenz.gen_dca_res'),
            RunStep('DCCA', 'dcca_env', 'scripts.experiments.lorenz.gen_dcca_res'),
            RunStep('Sh-Rec', 'shrec_env', 'scripts.experiments.lorenz.gen_shrec_res'),
            RunStep('Random', 'maco_env', 'scripts.experiments.lorenz.gen_random_res'),
            RunStep('SFA', 'sfa_env', 'scripts.experiments.lorenz.gen_sfa_res'),
            RunStep('MaCo', 'maco_env', 'scripts.experiments.lorenz.gen_maco_res'),
        ),
        combine_step=RunStep('Combine', 'maco_env', 'scripts.experiments.lorenz.Z_combine_final_res'),
    ),
    'lorenz_htune': FamilySpec(
        key='lorenz_htune',
        title='Lorenz Hyperparameter Tuning',
        combined_csv='htune.csv',
        methods=(
            MethodResult('PCA', 'pca_htune.csv'),
            MethodResult('kPCA', 'kpca_htune.csv'),
            MethodResult('ICA', 'ica_htune.csv'),
            MethodResult('DCA', 'dca_htune.csv'),
            MethodResult('SFA', 'sfa_htune.csv'),
        ),
        run_steps=(
            RunStep('PCA', 'maco_env', 'scripts.experiments.lorenz_hypertune.genres_pca_htune'),
            RunStep('KPCA', 'maco_env', 'scripts.experiments.lorenz_hypertune.genres_kpca_htune'),
            RunStep('ICA', 'maco_env', 'scripts.experiments.lorenz_hypertune.genres_ica_htune'),
            RunStep('DCA', 'dca_env', 'scripts.experiments.lorenz_hypertune.genres_dca_htune'),
            RunStep('SFA', 'sfa_env', 'scripts.experiments.lorenz_hypertune.genres_sfa_htune'),
        ),
        combine_step=RunStep('Combine', 'maco_env', 'scripts.experiments.lorenz_hypertune.genres_final_htune'),
    ),
    'example_logmap': FamilySpec(
        key='example_logmap',
        title='Example Logistic Map',
        combined_csv='mappercoach_res.csv',
        methods=(
            MethodResult('MaCo', 'mappercoach_res.csv'),
        ),
        run_steps=(
            RunStep('MaCo', 'maco_env', 'scripts.experiments.example_logmap.gen_exampleResults'),
        ),
        combine_step=RunStep('Combine', 'maco_env', 'scripts.experiments.example_logmap.Z_combine_final_res'),
    ),
    'noise_length': FamilySpec(
        key='noise_length',
        title='Noise and Length Sensitivity',
        combined_csv='noise_length_res.csv',
        methods=(
            MethodResult('Length Sweep', 'length_maco_res.csv'),
            MethodResult('Noise Sweep', 'noise_maco_res.csv'),
        ),
        run_steps=(
            RunStep('Length Sweep', 'maco_env', 'scripts.experiments.noise_length.maco_length'),
            RunStep('Noise Sweep', 'maco_env', 'scripts.experiments.noise_length.maco_noise'),
        ),
        combine_step=RunStep('Combine', 'maco_env', 'scripts.experiments.noise_length.Z_combine_final_res'),
    ),
    'dummy_experiment': FamilySpec(
        key='dummy_experiment',
        title='Dummy Import Mirror (Logmaps)',
        combined_csv='dummy_experiment_res.csv',
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
            RunStep('ICA', 'maco_env', 'scripts.experiments.dummy_experiment.gen_ica_res'),
            RunStep('PCA', 'maco_env', 'scripts.experiments.dummy_experiment.gen_pca_res'),
            RunStep('KPCA', 'maco_env', 'scripts.experiments.dummy_experiment.gen_kpca_res'),
            RunStep('CCA', 'maco_env', 'scripts.experiments.dummy_experiment.gen_cca_res'),
            RunStep('DCA', 'dca_env', 'scripts.experiments.dummy_experiment.gen_dca_res'),
            RunStep('DCCA', 'dcca_env', 'scripts.experiments.dummy_experiment.gen_dcca_res'),
            RunStep('Sh-Rec', 'maco_env', 'scripts.experiments.dummy_experiment.gen_shrec_res'),
            RunStep('Random', 'maco_env', 'scripts.experiments.dummy_experiment.gen_random_res'),
            RunStep('SFA', 'sfa_env', 'scripts.experiments.dummy_experiment.gen_sfa_res'),
            RunStep('MaCo', 'maco_env', 'scripts.experiments.dummy_experiment.gen_maco_res'),
            RunStep('AniSOM', 'maco_env', 'scripts.experiments.dummy_experiment.gen_anisom_res'),
        ),
        combine_step=RunStep('Combine', 'maco_env', 'scripts.experiments.dummy_experiment.Z_combine_final_res'),
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