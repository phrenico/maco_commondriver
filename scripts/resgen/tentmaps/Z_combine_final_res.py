from scripts.resgen.combine_utils import combine_result_files
from scripts.resgen.experiment_registry import get_family_spec
from scripts.resgen.tentmaps.config_tentmapres import interim_res_path, final_res_path

family_spec = get_family_spec('tentmaps')

df = combine_result_files(interim_res_path,
                          final_res_path / family_spec.combined_csv,
                          family_spec.result_files)

print(df.head())