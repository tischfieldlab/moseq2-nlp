import subprocess
import os
from moseq2_nlp.visualize import make_wordcloud
from moseq2_nlp.data import make_phrases_dataset

threshes = [.05,.05,.05]
n = 3
min_count = 3
visualize=True
max_plot=15
base_dir = '/cs/labs/mornitzan/ricci/data/abraira'
wordcloud_base_dir = './tourette_wordclouds'
save_base_dir = './tourette_phrases'

model_files = [os.path.join(base_dir, '2019-07-12_Junbing_Bicuculline_Celsr3-PvCre/first_model.p'),
               os.path.join(base_dir, '2020-11-10_Celsr3_R774H/robust_septrans_model_1000.p'),
               os.path.join(base_dir, '2021-04-02_Gad2Cre_Celsr3CKO/rST_model_1000.p'),
               os.path.join(base_dir, '2021-12-10_Celsr3_1894G/rST_model_1000.p'),
               os.path.join(base_dir, 'SstCre_Celsr3CKO/robust_septrans_model_20min_1000.p'),
               os.path.join(base_dir, 'WWC1_W88C/robust_septrans_model_500.p')]

index_files = [os.path.join(base_dir, '2019-07-12_Junbing_Bicuculline_Celsr3-PvCre/moseq2-index-cohort-sex-role.yaml'),
               os.path.join(base_dir, '2020-11-10_Celsr3_R774H/gender-genotype-index.yaml'),
               os.path.join(base_dir, '2021-04-02_Gad2Cre_Celsr3CKO/moseq2-index.sex-genotype.yaml'),
               os.path.join(base_dir, '2021-12-10_Celsr3_1894G/moseq2-index.sex-genotype.yaml'),
               os.path.join(base_dir, 'SstCre_Celsr3CKO/moseq2-index.sex-genotype.20min.yaml'),
               os.path.join(base_dir, 'WWC1_W88C/moseq2-index.sex-genotype.yaml')]

for m, (model_file, index_file) in enumerate(zip(model_files, index_files)):
    exp_name = model_file.split('/')[7]

    save_path      = os.path.join(save_base_dir, exp_name)
    #save_path      = os.path.join(save_path, exp_name + '.pickle')
    wordcloud_path = os.path.join(wordcloud_base_dir, exp_name)
    #make_wordcloud(save_path, wordcloud_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(wordcloud_path):
        os.makedirs(wordcloud_path)

    save_path      = os.path.join(save_path, exp_name + '.pickle')

    process_str = f'moseq2-nlp make-phrases {model_file} {index_file} --save-path {save_path} --n {n} --min-count {min_count} --visualize --wordcloud-path {wordcloud_path} --max-plot {max_plot} ' + ' '.join([f'--threshes {thresh}' for thresh in threshes]) + '&'

    subprocess.call(process_str, shell=True)
print('done')
