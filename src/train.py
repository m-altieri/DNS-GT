import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt
from colorama import Fore, Style

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from models import DELM, Word2Vec
from utils.distribute import DummyStrategy
from utils.sequencing import get_clusters_from_timestamp, pad


def config_gpus(args):
    if isinstance(args.gpu, int):
        device = tf.config.list_physical_devices("GPU")[args.gpu]
        tf.config.set_visible_devices(device, "GPU")
        print(f"Set {device} as the only visible device.")
    for device in tf.config.get_visible_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print("Cannot enable memory growth on device:", device)
            sys.exit(e)


def get_logger(verbose=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if verbose:
        logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def build_model(model, args, **kwargs):
    loss_reduction = tf.keras.losses.Reduction.NONE if args.distribute else "auto"
    if args.finetune:
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=loss_reduction
        )
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=loss_reduction
        )
    with kwargs["dist_strategy"].scope():
        if model.lower() == "delm":
            model = DELM(vars(args) | kwargs)
        elif model.lower() == "w2v":
            model = Word2Vec(vars(args) | kwargs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=loss,
            metrics=[],
            run_eagerly=args.eager,
        )
        return model


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", action="store", default="DELM")
    argparser.add_argument(
        "--es",
        action="store_true",
        help="Early Stopping",
    )
    argparser.add_argument(
        "--load",
        action="store",
        nargs="?",
        const="last",
        help="Whether to load the model from a saved checkpoint or to reinitialize a new one. Set it with no value or with 'last' to load the most recent checkpoint, or manually set a checkpoint file name.",
    )
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log debug information",
    )
    argparser.add_argument(
        "--epochs",
        action="store",
        default=5,
        type=int,
        help="Number of training epochs",
    )
    argparser.add_argument(
        "--bs", action="store", default=512, type=int, help="Batch size"
    )
    argparser.add_argument(
        "--lr", action="store", default=1e-4, type=float, help="Learning rate"
    )
    argparser.add_argument(
        "--demo",
        action="store_true",
        help="Used for debugging purposes",
    )
    argparser.add_argument(
        "--test-seq",
        action="store",
        type=int,
        help="Used for debugging purposes; choose the test sequence index",
    )
    argparser.add_argument(
        "--seqlen", action="store", default=32, type=int, help="Maximum sequence length"
    )
    argparser.add_argument(
        "--stride",
        action="store",
        default=1,
        type=int,
        help="Stride between sequences (how many queries to shift by)",
    )
    argparser.add_argument(
        "--include-start",
        action="store_true",
        help="Whether to include <START> as the first token of each sequence (total length is unaffected)",
    )
    argparser.add_argument(
        "--version",
        action="store",
        choices=["small", "all", "clean"],  # TODO clean should become the normal one
        default="small",
        help="Version of the dataset used.",
    )
    argparser.add_argument(
        "--tiny",
        action="store_true",
        help="Use for debugging purposes, to use a tiny portion of the dataset to get faster feedback.",
    )
    argparser.add_argument(
        "--gpu",
        action="store",
        help="If it is an integer (eg. --gpu 3), run on a single specific GPU. "
        + "If it is an array (eg. [2,4]), distribute the execution on the specified GPUs. "  # TODO
        + "If it is `all`, distribute on all GPUs.",
    )
    argparser.add_argument("--tensorboard", "--tb", action="store_true")
    argparser.add_argument(
        "--quick-tb",
        action="store_true",
        help="Whether to reutilize the same TensorBoard folder. Allows for quicker debugging.",
    )
    argparser.add_argument("--eager", action="store_true")
    argparser.add_argument("--blocks", action="store", type=int)
    argparser.add_argument("--group-hosts", action="store_true")
    argparser.add_argument(
        "--run-name",
        action="store",
        default=f'model-{time.strftime("%y%m%d-%H%M%S", time.localtime())}',
        help="Name used when saving to file. Has no effect if --load.",
    )
    argparser.add_argument("--omega", action="store", type=float)
    argparser.add_argument("--shuffle", action="store_true")
    argparser.add_argument(
        "--type",
        action="store",
        help="Model type. It is used by model classes that have multiple subtypes, like Word2Vec.",
    )
    argparser.add_argument(
        "--dim",
        action="store",
        type=int,
        help="Dimension of the embeddings. Now host and domain embeddings always have the same dimension.",
    )
    argparser.add_argument(
        "--finetune",
        "--ft",
        action="store_true",
        help="Use the classification workflow instead of the embedding learning workflow.",
    )
    argparser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze all layers except the classification ones. Only has effect if --finetune.",
    )
    argparser.add_argument(
        "--test-fold",
        type=int,
        help="Test fold to choose during finetuning. "
        + "Domains contained in the fold will not be used for loss computation, making them suitable for testing. "
        + "If not set, all domains will be considered during loss computation. "
        + "Only has effect if --finetune.",  # TODO Now it is mandatory if --finetune, otherwise there are errors. Make it optional
    )
    argparser.add_argument(
        "--from-pretrained",
        "--from-pt",
        action="store_true",
        help="Whether to load weights from existing finetuned model or from pretrained model. "
        + "Only has effect if --finetune.",
    )
    argparser.add_argument("--max-tokens", action="store", type=int)
    argparser.add_argument("--concat-hosts", action="store_true")

    args = argparser.parse_args()

    assert args.test_seq is None or args.test_seq > 0

    try:
        args.gpu = int(args.gpu)
    except:  # it is not a number, it's either None or `all`
        pass
    try:
        if "[" in args.gpu:  # if it is a list
            args.gpu = [int(i) for i in args.gpu.strip("[").strip("]").split(",")]
    except:  # it is not a list, let's try with a number
        pass
    if isinstance(args.gpu, list) or args.gpu == "all":
        args.distribute = True
        assert tf.config.get_visible_devices("GPU") == tf.config.list_physical_devices(
            "GPU"
        )  # if distribute, devices cannot be set as not visible, to avoid possible bugs
    else:
        args.distribute = False

    if args.demo:
        args.eager = True
        args.tensorboard = True
        args.gpu = None
        args.distribute = False
        args.bs = 1
    return args


def seq_generator_from_folder(input_folder: str, **kwargs) -> iter:
    """Create a generator for the tf.data API.

    Args:
        input_folder (str): folder containing .npy files, each representing a matrix of shape (n_queries, 2).

    Keyword args:
        seqlen (int): maximum sequence length
        strategy (str): sequencing strategy. If "cluster", sequences will be created by splitting
        queries based on their timestamp, resulting in sequences of different lengths being padded.
        Each query will appear in a single sequence.
        If "fixed", sequences will be cut at exactly seqlen, disregarding the timestamp.
        Each query will appear in multiple sequences, depending on the stride value.
        stride (int): if strategy is "fixed", by how many queries to shift after each sequence
        include_start (bool): Deprecated. whether to include a <START> token at the beginning of the sequence
        include_class (bool): used in finetuning to include the class label of each query
        group_hosts (bool): whether each sequence should have only queries from the same host
        model (str): model to use. Either "delm" or "w2v"
        vocab (str): domains vocabulary. Only used in finetuning
        tiny_amount (bool): Deprecated. Whether to use a small number of queries for debugging purposes
        verbose (bool): whether to print additional debugging info

    Yields:
        iter: a generator that yields input sequences for the model
    """

    for f in os.listdir(input_folder):
        if os.path.splitext(os.path.join(input_folder, f))[-1] != ".npy":
            continue

        seqs = _create_sequences(os.path.join(input_folder, f), **kwargs)
        for seq in seqs:
            yield seq


def _create_sequences(input_file: str, **kwargs):
    queries = np.load(input_file, allow_pickle=True)
    if kwargs.get("tiny_amount"):
        queries = queries[:10000]
    if kwargs.get("include_class"):
        labels = pd.read_csv(
            os.path.join("scripts", "labels.csv"), index_col=0, header=[0, 1]
        )
        labels.columns = pd.MultiIndex.from_tuples(
            [
                ("domain", ""),
                ("advertising", "good"),
                ("advertising", "ok"),
                ("malicious", "good"),
                ("malicious", "ok"),
                ("suspicious", "good"),
                ("suspicious", "ok"),
                ("tracking", "good"),
                ("tracking", "ok"),
                ("other", "good"),
                ("other", "ok"),
                ("any", "good"),
                ("any", "ok"),
            ]
        )
        # Only use labels for domains in embs (i.e. in the vocabulary)
        labels = labels[labels["domain"].isin(kwargs.get("vocab"))]
        labels = labels.reset_index()

        # take (any, ok) column
        labels = labels[["domain", "any"]]
        labels = labels.to_numpy()[:, [0, 2]]

        # add class to each query
        sorter = np.argsort(labels[:, 0])
        idx = sorter[np.searchsorted(labels[:, 0], queries[:, 1], sorter=sorter)]
        classes = labels[idx, 1]
        queries = np.concatenate(
            [queries, np.expand_dims(classes, -1).astype(str)], axis=-1
        )
    if kwargs.get("group_hosts"):  # sort queries by host, preserving row structure
        queries = queries[np.lexsort((queries[:, -1], queries[:, 0]))]
    seqs = []
    if kwargs.get("model") == "delm":
        if kwargs.get("strategy") == "cluster":
            for host in np.unique(queries[:, 0]):  # for each unique host

                host_queries = queries[
                    np.where(queries[:, 0] == host)[0]
                ]  # get queries made by the current host

                # get cluster labels of that host's queries
                host_cluster_labels = get_clusters_from_timestamp(host_queries)

                # from each of those clusters we are going to make a sequence
                for c in range(np.max(host_cluster_labels) + 1):
                    cluster = host_queries[
                        np.where(host_cluster_labels == c)
                    ]  # get queries associated to the current cluster label
                    cluster = cluster[
                        :, :-1
                    ]  # remove timestamp once it's no longer needed

                    # take first element of the domain (the domain is a list with always one token)
                    # TODO this is only ok for the trivial tokenizer
                    cluster = [[q[0], q[1][0]] for q in cluster]

                    # truncate clusters (sequences) longer than seqlen, moving the excess to a new sequence
                    while len(cluster) > kwargs.get("seqlen"):
                        if kwargs.get("verbose"):
                            print(
                                f"[INFO] Truncating sequence with cluster ID {c} for host {host}: length of {len(cluster)} exceeds seqlen of {kwargs.get('seqlen')}."
                            )
                        truncated_cluster = cluster[: kwargs.get("seqlen")]
                        seqs.append(truncated_cluster)
                        cluster = cluster[kwargs.get("seqlen") :]

                    seqs.append(
                        pad(cluster, kwargs.get("seqlen"))
                    )  # this sequence will not be full, so we have to pad it to seqlen
            seqs = np.array(seqs, dtype=object)
        elif (
            kwargs.get("strategy") == "fixed"
        ):  # output [queries - stride, seqlen, 2 or 3]
            actual_seqlen = kwargs.get("seqlen") - kwargs.get("include_start")
            seqs = np.empty(
                shape=(
                    (len(queries) - actual_seqlen) // kwargs.get("stride") + 1,
                    kwargs.get("seqlen"),
                    3 if kwargs.get("include_class") else 2,
                ),
                dtype=object,
            )
            for i, _ in enumerate(seqs):
                if kwargs.get("include_start"):
                    seqs[i][0] = ["<START>", "<START>"]
                seqs[i][kwargs.get("include_start") :] = queries[
                    i * kwargs.get("stride") : i * kwargs.get("stride") + actual_seqlen
                ]
        else:
            raise ValueError("'strategy' kwarg must be either 'cluster' or 'fixed'.")
    elif kwargs.get("model") == "w2v":  # output [queries, seqlen]
        seqs = np.array(Word2Vec.create_pairs(queries[:, 1:], kwargs.get("seqlen")))
    else:
        raise ValueError("Specify model to create sequences.")

    return seqs


def find_last_checkpoint(dir):
    if len(os.listdir(dir)) > 0:
        checkpoint = os.listdir(dir)[
            [os.path.getmtime(os.path.join(dir, f)) for f in os.listdir(dir)].index(
                max([os.path.getmtime(os.path.join(dir, f)) for f in os.listdir(dir)])
            )
        ]
    else:
        checkpoint = ""
    return checkpoint


def default_checkpoint(args):
    return f"{args.run_name}.h5"


def indent(depth=1):
    return f"".join(["--" for i in range(depth - 1)]) + "> "


def main():
    args = parse_args()

    logger = get_logger(args.verbose)
    logger.info("Started training with args:")
    logger.info("\n".join([f"{indent(1)}{k}: {vars(args)[k]}" for k in vars(args)]))

    args.version = "clean"
    path = "/mnt/storage15/TI-2016/npy/tokenized/trivial"
    queries_path = path
    domains_vocab_path = os.path.join(path, "domain_vocab.txt")
    domains_vocab_path = os.path.join(path, "host_vocab.txt")

    # queries_path = {"clean": "/mnt/storage15/TI-2016/npy"}[args.version]
    # logger.info(f"{Fore.YELLOW}Using --version {args.version}{Fore.RESET}")
    # domains_vocab_path = f"../data/vocabs/{args.version}/domains_vocab.txt"
    # hosts_vocab_path = f"../data/vocabs/{args.version}/hosts_vocab.txt"
    with open(domains_vocab_path, "r") as f:
        domains_vocab = [l.strip() for l in f.readlines()]

    config_gpus(args)

    train = tf.data.Dataset.from_generator(
        lambda: seq_generator_from_folder(
            os.path.join(queries_path, "train"),
            seqlen=args.seqlen,
            strategy="cluster",
            stride=args.stride,
            include_start=args.include_start,
            include_class=args.finetune,
            group_hosts=args.group_hosts,
            model=args.model.lower(),
            vocab=domains_vocab,
            tiny_amount=args.tiny,
            verbose=False,
        ),
        output_signature=tf.TensorSpec(
            shape=(args.seqlen, 2 + args.finetune), dtype=tf.string
        )
        if args.model.lower() == "delm"
        else tf.TensorSpec(shape=(args.seqlen, 1 + args.finetune), dtype=tf.string),
    )
    test = tf.data.Dataset.from_generator(
        lambda: seq_generator_from_folder(
            os.path.join(queries_path, "test"),
            seqlen=args.seqlen,
            strategy="cluster",
            stride=args.stride,
            include_start=args.include_start,
            include_class=args.finetune,
            group_hosts=args.group_hosts,
            model=args.model.lower(),
            vocab=domains_vocab,
            tiny_amount=args.tiny,
            verbose=False,
        ),
        # output_signature=tf.TensorSpec(
        #     shape=(args.seqlen, 2 + args.finetune), dtype=tf.string
        # )
        output_signature=tf.TensorSpec(shape=[None, 2 + args.finetune], dtype=tf.string)
        if args.model.lower() == "delm"
        else tf.TensorSpec(shape=(args.seqlen, 1 + args.finetune), dtype=tf.string),
    )
    if not args.demo and args.shuffle:
        train = train.shuffle(1000000)
    train = train.batch(args.bs).prefetch(tf.data.AUTOTUNE)
    test = test.batch(args.bs).prefetch(tf.data.AUTOTUNE)

    # Distribution
    dist_strategy = None
    if args.distribute:
        gpus = (
            [f"/gpu:{i}" for i in args.gpu] if isinstance(args.gpu, list) else None
        )  # initializing MirroredStrategy with None uses all gpus
        dist_strategy = tf.distribute.MirroredStrategy(gpus)
        logger.warning(
            f"{Fore.BLUE}Distributing on {dist_strategy.num_replicas_in_sync} devices.{Style.RESET_ALL}"
        )

        # Data Config
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        train = train.with_options(options)
        test = test.with_options(options)
        train = dist_strategy.experimental_distribute_dataset(train)
        test = dist_strategy.experimental_distribute_dataset(test)
    else:
        dist_strategy = DummyStrategy

    # Build Model
    model = build_model(
        args.model,
        args,
        dist_strategy=dist_strategy,
    )
    if args.finetune:  # freeze all layers but the last classification layer
        model.finetune()
    else:  # unfreeze in case it was frozen
        model.pretrain()

    # Manage checkpoint
    if not os.path.exists("../checkpoints"):
        os.makedirs("../checkpoints")
    checkpoint_folder = os.path.join(
        "../checkpoints", f"{args.model}{f'-{args.type}' if args.type else ''}"
    )
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_name = default_checkpoint(args)

    logger.info(f"Calling model to initialize layers...")
    if args.distribute:
        model.distributed_test_step(next(iter(test)))
    else:
        model.test_step(next(iter(test)))

    if args.load:  # load saved weights if --load
        checkpoint_name = (
            find_last_checkpoint(dir=checkpoint_folder)
            if args.load == "last"
            else args.load
        )
        load_weights_path = os.path.join(
            checkpoint_folder,
            f"{os.path.splitext(checkpoint_name)[0]}{f'.finetuned-{args.test_fold}' * (args.finetune and not args.from_pretrained)}.h5",
        )

        logger.info(f"Trying to load weights from {load_weights_path}...")
        try:
            with dist_strategy.scope():  # not sure if the scope is needed
                model.load_weights(
                    load_weights_path,
                    skip_mismatch=False,  # let's try false, it was true
                    by_name=True,
                )
            logger.info(f"Model weights loaded from {load_weights_path}.")
        except Exception as e:
            logger.error(
                f"{Fore.YELLOW}Exception when trying to load checkpoint {load_weights_path}:\n{Style.DIM}{e}"
                + f"\n{Style.NORMAL}Continuing without loading checkpoint.{Style.RESET_ALL}"
            )
            checkpoint_name = default_checkpoint(args)

    if args.demo:
        logger.info(
            f"{Style.BRIGHT}\nDomain Embeddings Language Model{Style.RESET_ALL}\n"
            + "Please refer to https://gitlab.jrc.ec.europa.eu/jrc-projects/createg/cdp-bari/dns/-/tree/main/ for roadmap and updates.\n"
            + "Syntax: <Host> <Domain> -> <Predicted Domain> (<prob%>) [(<Unmasked Domain> <prob%>)]\n"
        )
        seq_idx = args.test_seq or np.random.randint(0, 1000)
        seqs = (
            train.unbatch().shuffle(seq_idx).take(5).as_numpy_iterator()
        )  # it was skip(seq_idx) instead of shuffle(seq_idx)
        seqs = np.array([s for s in seqs], dtype=object)
        for s in range(len(seqs)):
            seq = seqs[s : s + 1]

            mask = np.zeros_like(seq)
            # place 1's where you want to replace tokens with <MASK>
            # axis 0 is always 0 (array of length 1), axis 1 is the index of token within the sequence, axis 2 is 0 for host and 1 for domain
            # example: mask[0, 1, 1]
            #   always zero ^  ^  ^
            #     second token |  |
            #                     | domain
            mask[0, 0, -1] = 1
            mask[0, 1, -1] = 1
            mask[0, 2, -1] = 1
            masked_seq = np.where(mask, np.full_like(seq, "<MASK>", dtype=object), seq)

            pred, loss, kwout = model._predict(seq, mask)
            print(f"Seq index: {seq_idx}")

            if args.finetune:
                pred = np.array(pred).flatten()
                for p, _ in enumerate(pred):
                    domain = seq[0, p, 1]
                    if type(domain) is bytes:
                        domain = domain.decode("utf-8")
                    label = seq[0, p, 2]
                    if type(label) is bytes:
                        label = label.decode("utf-8")
                    print(
                        f"{Fore.CYAN if kwout.get('in_fold')[p] else ''}{domain} ({label}) -> {pred[p]:.3f}{Style.RESET_ALL}"
                    )
                print(f"{Style.BRIGHT}Loss: {loss:.3f}{Style.RESET_ALL}")
            else:
                pred = pred[0]

                for i in range(len(pred)):
                    masked_host = model.inverse_hosts_lookup(
                        model.hosts_lookup(masked_seq[0, i, 0])
                    )  # I am actually interested in what token the model considers, not what we pass as input (if the token is not in the vocabulary, it will be treated as <UNK>)
                    if type(masked_host) is bytes:
                        masked_host = masked_host.decode("utf-8")

                    domain = model.inverse_domains_lookup(
                        model.domains_lookup(seq[0, i, 1])
                    )
                    if type(domain) is bytes:
                        domain = domain.decode("utf-8")

                    masked_domain = model.inverse_domains_lookup(
                        model.domains_lookup(masked_seq[0, i, 1])
                    )
                    if type(masked_domain) is bytes:
                        masked_domain = masked_domain.decode("utf-8")

                    predicted_token = model.inverse_domains_lookup(
                        np.array(pred).argmax(axis=-1)[i]
                    )
                    domain_index = model.domains_lookup(domain)
                    logger.info(
                        f"{masked_host} {masked_domain} -> {f'{Fore.GREEN}' if domain == predicted_token else f'{Fore.RED}'}{predicted_token} ({100*(np.array(pred).max(axis=-1)[i]):.2f}%){Style.RESET_ALL} {f'{Style.DIM}({domain} {100*(np.array(pred)[i,domain_index]):.2f}%) {Style.RESET_ALL}' if not domain == predicted_token else ''}"
                    )
                logger.info(f"{Style.BRIGHT}Loss: {loss:.3f}{Style.RESET_ALL}")

        sys.exit(0)

    if args.verbose:
        logger.info(model.summary())

    # Save model weights
    save_weights_path = os.path.join(
        checkpoint_folder,
        f"{os.path.splitext(checkpoint_name)[0]}{f'.finetuned-{args.test_fold}' * args.finetune}.h5",
    )

    logger.info("Starting model training...")
    if not args.distribute:
        model.fit(
            x=train,
            y=None,
            validation_data=test,
            validation_freq=1,
            batch_size=args.bs,
            epochs=args.epochs,
            callbacks=[
                ModelCheckpoint(
                    save_weights_path,
                    monitor="loss",
                    save_weights_only=True,
                ),
            ],
        )
    else:  # if args.distribute
        num_batches = None
        for epoch in range(args.epochs):
            total_loss = 0.0
            pbar = tqdm(train, total=num_batches)

            # Train loop
            current_batch = 0
            for x in pbar:
                total_loss += model.distributed_train_step(x)
                current_batch += 1
                pbar.set_description(
                    f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {total_loss / current_batch:.4f}"
                )

            # Test loop
            for x in test:
                total_loss += model.distributed_test_step(x)
            logger.info(f"Test Loss: {total_loss / current_batch:.4f}")

            # Save model weights
            with dist_strategy.scope():
                model.save_weights(save_weights_path)

            num_batches = current_batch

    logger.info(f"Model training completed.")

    with dist_strategy.scope():  # not sure if the scope is needed
        model.save_weights(save_weights_path)

    # Save embeddings
    logger.info("Saving embeddings...")
    if not os.path.exists("../embeddings"):
        os.makedirs("../embeddings")
    embeddings_folder = os.path.join(
        "../embeddings", f"{args.model}{f'-{args.type}' if args.type else ''}"
    )
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)
    domain_embeddings = (
        model.domain_embeddings.embeddings.numpy()
    )  # TODO may break if model class uses a different variable name; use a get_embeddings() function instead
    np.save(
        os.path.join(
            embeddings_folder,
            f"emb-{os.path.splitext(checkpoint_name)[0]}{f'.finetuned-{args.test_fold}' * args.finetune}.npy",
        ),
        domain_embeddings,
    )

    # Save predictions
    logger.info("Saving model predictions...")
    if not args.finetune:
        raise ValueError(
            f"{Fore.YELLOW}Saving model predictions can only be done in --finetune.{Style.RESET_ALL}"
        )
    elif args.distribute:
        raise NotImplementedError(
            f"{Fore.YELLOW}Saving model predictions is not supported in --distribute.{Style.RESET_ALL}"
        )  # TODO implement saving model predictions in --distribute

    else:
        # ISSUE should I evaluate on train, test or both? consider that in any case the model is never trained
        # on in_fold domains, even on train. on the other hand, one may argue it's easier if the model has already
        # seen that sequence. decide
        # TODO make this algorithm more efficient, and possibly refactor it out
        num_batches = sum([1 for _ in test])
        d, trues, preds = (
            np.zeros((num_batches * args.bs * args.seqlen), dtype=object),
            np.zeros((num_batches * args.bs * args.seqlen)),
            np.zeros((num_batches * args.bs * args.seqlen)),
        )
        batch_idx = 0
        for x in tqdm(test):
            domains = (
                x[..., 1] if args.model == "DELM" else x[:, args.seqlen // 2, 0]
            )  # (B,L) or (B,)
            true = (
                x[..., -1] if args.model == "DELM" else x[:, args.seqlen // 2, 1]
            )  # (B,L) or (B,)
            pred, _, kwout = model._predict(
                x
            )  # ( (B,L), (), (B,L) ) or ( (B,), (), (B,) )

            domains_per_seq = (
                args.seqlen
                if args.model == "DELM"
                else 1
                if args.type == "CBOW"
                else args.seqlen - 1
            )
            d[
                batch_idx
                * args.bs
                * domains_per_seq : batch_idx
                * args.bs
                * domains_per_seq
                + len(domains[kwout["in_fold"]])
            ] = domains[kwout["in_fold"]]
            trues[
                batch_idx
                * args.bs
                * domains_per_seq : batch_idx
                * args.bs
                * domains_per_seq
                + len(true[kwout["in_fold"]])
            ] = true[kwout["in_fold"]]
            preds[
                batch_idx
                * args.bs
                * domains_per_seq : batch_idx
                * args.bs
                * domains_per_seq
                + len(pred[kwout["in_fold"]])
            ] = pred[kwout["in_fold"]]

            batch_idx += 1
        df = pd.DataFrame({"domains": d, "true": trues, "pred": preds})

        df = df[df["domains"].notnull()]  # should be useless
        df = df[df["domains"] != ""]  # should be useless
        df = df[df["domains"] != 0]

        predictions_path = f"../predictions/{args.model}/"
        if args.type is not None:
            predictions_path += f"{args.type}/"
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        df.to_csv(os.path.join(predictions_path, f"preds-fold{args.test_fold}.csv"))

        # TODO I DON'T WANT TO COMPUTE METRICS HERE, IT SHOULD BE SEPARATE; HERE I ONLY SAVE PREDICTIONS

        df["pred_hard"] = df["pred"].round()
        # df["tp"] = np.logical_and(df["true"] == 1, df["pred_hard"] == 1)
        # df["fp"] = np.logical_and(df["true"] == 0, df["pred_hard"] == 1)
        # df["fn"] = np.logical_and(df["true"] == 1, df["pred_hard"] == 0)
        # df["tn"] = np.logical_and(df["true"] == 0, df["pred_hard"] == 0)

        # logger.info(
        #     f"TP: {df['tp'].sum()}\nFP: {df['fp'].sum()}\nFN: {df['fn'].sum()}\nTN: {df['tn'].sum()}"
        # )

        logger.info(
            f"SKlearn Confusion matrix:\n{sklearn.metrics.confusion_matrix(df['true'], df['pred_hard'], labels=[1,0])}"
        )
        auc = sklearn.metrics.roc_auc_score(df["true"], df["pred"])
        fpr, tpr, _ = sklearn.metrics.roc_curve(df["true"], df["pred"])
        logger.info(f"AUC: {auc}")
        fig = plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, label=f"ROC fold {args.test_fold} (AUC = {auc:.2f})")
        plt.legend(loc="lower right")
        plt.savefig(
            os.path.join(
                predictions_path,
                f"roc-{args.model}{f'-{args.type}' if args.type else ''}{f'-{args.test_fold}'}.png",
            )
        )
        # logger.info(f"AUC: {sklearn.metrics.auc(fpr, tpr)}")
        # display = sklearn.metrics.RocCurveDisplay.from_predictions(
        #     df["true"], df["pred"]
        # )
        # print(thresholds)
        # print(thresholds.shape)
        # display.plot()
        # plt.savefig("sklearnauc.png")
        # plt.clf()

        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        # ax1.plot(fpr)
        # ax2.plot(tpr)
        # fig.savefig("roc.png")

    logger.info("Model predictions saved.")


if __name__ == "__main__":
    main()
