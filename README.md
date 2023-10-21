# On the Relationship Between Relevance and Conflict in Online Social Link Recommendations (NeurIPS 2023)
Yanbang Wang and Jon Kleinberg, Cornell University

## Environment
* `python=3.10.4`
* `numpy=1.23.1`
* `sklearn=0.24`
* `networkx=2.8.4`
* `torch=2.8.4`
* `torch-geometric=2.3.0`
* `cvxpy=1.2.2`
* `tqdm=4.64.1`

## Running the Code
`python main.py --dataset <dataset> --n <negative sampling rate>`
* `<dataset>` can be `reddit` or `twitter`
* `<negative sampling rate>` should be an integer in [1, 9]

