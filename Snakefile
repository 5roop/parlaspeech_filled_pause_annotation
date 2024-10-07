# Run with
# export CUDA_VISIBLE_DEVICES=4; mamba activate fpval; snakemake -j 2 -k --use-conda --batch gather=1/5
# and to catch the last files that escaped batching:
# sleep 10000;export CUDA_VISIBLE_DEVICES=3; while true; do sleep 600; snakemake -j 1 -k --use-conda; done;


from pathlib import Path
def has_mp3s(p: Path) -> bool:
    mp3s = list(p.glob("*.mp3"))
    return mp3s != []
hr_folders = [i for i in Path("/cache/nikolal/parlaspeech-hr/audio/").glob("*/") if has_mp3s(i)]
rs_folders = [i for i in Path("/cache/nikolal/parlaspeech-rs/audio/").glob("*/") if has_mp3s(i)]

rule gather:
    input: expand("output/rs/{dirs}.jsonl", dirs=[i.name for i in rs_folders]) + expand("output/hr/{dirs}.jsonl", dirs=[i.name for i in hr_folders])

rule doone:
    input:
        input_dir = "/cache/nikolal/parlaspeech-{lang}/audio/{directory}"
    output:
        "output/{lang}/{directory}.jsonl"
    params:
        batchsize=1
    conda:
        "fpval.yml"
    script:
        "scripts/transcribe_file.py"
