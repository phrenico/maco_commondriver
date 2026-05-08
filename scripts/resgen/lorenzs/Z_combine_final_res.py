from scripts.resgen.combine_utils import combine_result_files
from scripts.resgen.experiment_registry import get_family_spec
from scripts.resgen.lorenzs.config_lorenzres import interim_res_path, final_res_path

print('Starting to combine results from ', interim_res_path)
print('Loading results from ', interim_res_path)

print("Combining Lorenz's results")
family_spec = get_family_spec('lorenzs')

df = combine_result_files(interim_res_path,
                          final_res_path / family_spec.combined_csv,
                          family_spec.result_files)

print("Saving combined results to ", final_res_path / family_spec.combined_csv)

print('Saved combined results')