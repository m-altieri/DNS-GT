import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from models import DELM, Word2Vec
from utils.distribute import DummyStrategy


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
    loss = (
        tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=loss_reduction
        )
        if not args.finetune
        else tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=loss_reduction
        )
    )
    with kwargs["dist_strategy"].scope():
        if model.lower() == "delm":
            model = DELM(
                seqlen=args.seqlen,
                blocks=args.blocks,
                tensorboard=args.tensorboard,
                quick_tb=args.quick_tb,
                run_name=args.run_name,
                omega=args.omega,
                version=args.version,
                dim=args.dim,
                bs=args.bs,
                dist_strategy=kwargs["dist_strategy"],
            )
        elif model.lower() == "w2v":
            model = Word2Vec(
                type=args.type,
                dim=args.dim,
                tensorboard=args.tensorboard,
                quick_tb=args.quick_tb,
                run_name=args.run_name,
            )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss=loss,
            metrics=[],
            run_eagerly=args.eager,
        )
        return model


def parse_args():
    argparser = argparse.ArgumentParser()
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
        "--bs", action="store", default=256, type=int, help="Batch size"
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
        choices=["small", "all"],
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
    argparser.add_argument("--blocks", action="store", type=int, default=4)
    argparser.add_argument("--group-hosts", action="store_true", default=True)
    argparser.add_argument(
        "--run-name",
        action="store",
        default=f'model-{time.strftime("%y%m%d-%H%M%S", time.localtime())}',
        help="Name used when saving to file. Has no effect if --load.",
    )
    argparser.add_argument("--omega", action="store", type=float, default=0.8)
    argparser.add_argument("--shuffle", action="store_true")
    argparser.add_argument("model", action="store", default="DELM")
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
        help="Freeze all layers and use the classification workflow instead of the embedding learning workflow.",
    )
    argparser.add_argument(
        "--from-pretrained",
        "--from-pt",
        action="store_true",
        help="Whether to load weights from existing finetuned model or from pretrained model. "
        + "Only has effect if --finetune.",
    )
    argparser.add_argument("--max-tokens", action="store", type=int)

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
    return args


def seq_generator_from_folder(
    input_folder,
    seqlen,
    stride=1,
    include_start=False,
    include_class=False,
    group_hosts=True,
    model=None,
    vocab=None,
    tiny_amount=None,
):
    """Folder containing .npy files, each representing a matrix of shape (n_queries, 2)."""
    for f in os.listdir(input_folder):
        if os.path.splitext(os.path.join(input_folder, f))[-1] != ".npy":
            continue
        seqs = create_sequences(
            os.path.join(input_folder, f),
            seqlen,
            stride,
            include_start,
            include_class,
            group_hosts,
            model,
            vocab,
            tiny_amount,
        )
        for seq in seqs:
            yield seq


def create_sequences(
    input_file,
    seqlen,
    stride=1,
    include_start=False,
    include_class=False,
    group_hosts=True,
    model=None,
    vocab=None,
    tiny_amount=None,
):

    queries = np.load(input_file, allow_pickle=True)

    if tiny_amount:
        queries = queries[:10000]

    if include_class:
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
        labels = labels[labels["domain"].isin(vocab)]
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

    if group_hosts:  # sort queries by host, preserving row structure
        queries = queries[np.argsort(queries[:, 0])]

    if model == "delm":  # output [queries - stride, seqlen, 2 or 3]
        actual_seqlen = seqlen - include_start
        seqs = np.empty(
            shape=(
                (len(queries) - actual_seqlen) // stride + 1,
                seqlen,
                3 if include_class else 2,
            ),
            dtype=object,
        )
        for i, _ in enumerate(seqs):
            if include_start:
                seqs[i][0] = ["<START>", "<START>"]
            seqs[i][include_start:] = queries[i * stride : i * stride + actual_seqlen]

    elif model == "w2v":  # output [queries, seqlen]
        seqs = np.array(Word2Vec.create_pairs(queries[:, 1:], seqlen))
        print(seqs)
    else:
        raise ValueError("Specify model to create sequences.")

    return seqs


def find_last_checkpoint(dir="checkpoints"):
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

    queries_path = f"preprocessing/arrays/{args.version}/queries/"
    domains_vocab_path = f"preprocessing/vocabs/{args.version}/domains_vocab.txt"
    hosts_vocab_path = f"preprocessing/vocabs/{args.version}/hosts_vocab.txt"

    with open(domains_vocab_path, "r") as f:
        domains_vocab = [l.strip() for l in f.readlines()]

    # config_tf(args)
    config_gpus(args)

    # Data Pipeline
    train = tf.data.Dataset.from_generator(
        lambda: seq_generator_from_folder(
            os.path.join(queries_path, "train"),
            stride=args.stride,
            seqlen=args.seqlen,
            include_start=args.include_start,
            include_class=args.finetune,
            group_hosts=args.group_hosts,
            model=args.model.lower(),
            vocab=domains_vocab,
            tiny_amount=args.tiny,
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
            stride=args.stride,
            seqlen=args.seqlen,
            include_start=args.include_start,
            include_class=args.finetune,
            group_hosts=args.group_hosts,
            model=args.model.lower(),
            vocab=domains_vocab,
            tiny_amount=args.tiny,
        ),
        output_signature=tf.TensorSpec(
            shape=(args.seqlen, 2 + args.finetune), dtype=tf.string
        )
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
        )  # setting None uses all gpus
        dist_strategy = tf.distribute.MirroredStrategy(gpus)
        print(
            f"{Fore.YELLOW}Distributing on {dist_strategy.num_replicas_in_sync} devices.{Style.RESET_ALL}"
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
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    checkpoint_folder = os.path.join(
        "checkpoints", f"{args.model}{f'-{args.type}' if args.type else ''}"
    )
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    checkpoint_name = default_checkpoint(args)

    # Load saved weights if --load
    if args.load:
        checkpoint_name = (
            find_last_checkpoint(dir=checkpoint_folder)
            if args.load == "last"
            else args.load
        )
        load_weights_path = os.path.join(
            checkpoint_folder,
            f"{os.path.splitext(checkpoint_name)[0]}{'.finetuned' * (args.finetune and not args.from_pretrained)}.h5",
        )
        logger.info(f"Trying to load weights from {load_weights_path}...")

        logger.info(f"Calling model to initialize layers...")
        if args.distribute:
            model.distributed_test_step(next(iter(test)))
        else:
            model.test_step(next(iter(test)))

        try:
            with dist_strategy.scope():  # not sure if the scope is needed
                model.load_weights(
                    load_weights_path,
                    skip_mismatch=True,
                    by_name=True,
                )
            logger.info(f"Model weights loaded from {load_weights_path}.")
        except Exception as e:
            logger.error(
                f"{Fore.RED}Exception when trying to load checkpoint {load_weights_path}:\n{Style.DIM}{e}"
                + f"\n{Style.NORMAL}Continuing without loading checkpoint.{Style.RESET_ALL}"
            )
            checkpoint_name = default_checkpoint(args)

    if args.demo:
        logger.info(
            f"{Style.BRIGHT}\nDomain Embeddings Language Model{Style.RESET_ALL}\n"
            + "Please refer to https://gitlab.jrc.ec.europa.eu/jrc-projects/createg/cdp-bari/dns/-/tree/main/ for roadmap and updates.\n"
            + "Syntax: <Host> <Domain> -> <Predicted Domain> (<prob%>) [(<Unmasked Domain> <prob%>)]\n"
        )
        seq = (
            train.unbatch()
            .skip(args.test_seq or np.random.randint(0, 1000))
            .take(1)
            .as_numpy_iterator()
        )
        seq = np.array([s for s in seq], dtype=object)

        # uncomment this assignment to manually create a sequence
        # note that arbitrarily created sequences will be harder to predict, since they don't follow any pattern found in the training data
        # seq = np.array(
        #     [
        #         [
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #             ["172.31.1.6", "graph.facebook.com"],
        #         ]
        #     ],
        #     dtype=object,
        # )

        mask = np.zeros_like(seq)
        # place 1's where you want to replace tokens with <MASK>
        # axis 0 is always 0 (array of length 1), axis 1 is the index of token within the sequence, axis 2 is 0 for host and 1 for domain
        # example: mask[0, 1, 1]
        #   always zero ^  ^  ^
        #     second token |  |
        #                     | domain
        mask[0, 1, 1] = 1

        masked_seq = np.where(mask, np.full_like(seq, "<MASK>", dtype=object), seq)

        pred, loss = model._predict(seq, mask)
        print(loss)
        print(np.shape(loss))
        pred = pred[0]

        for i in range(len(pred)):
            masked_host = masked_seq[0, i, 0]
            if type(masked_host) is bytes:
                masked_host = masked_host.decode("utf-8")
            domain = seq[0, i, 1]
            if type(domain) is bytes:
                domain = domain.decode("utf-8")
            masked_domain = masked_seq[0, i, 1]
            if type(masked_domain) is bytes:
                masked_domain = masked_domain.decode("utf-8")

            predicted_token = domains_vocab[np.array(pred).argmax(axis=-1)[i]]
            domain_index = domains_vocab.index(domain)
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
        f"{os.path.splitext(checkpoint_name)[0]}{'.finetuned' * args.finetune}.h5",
    )

    logger.debug("Starting model training...")
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
        for epoch in range(args.epochs):
            total_loss = 0.0
            num_batches = 0
            pbar = tqdm(
                train,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ''{rate_inv_fmt} {postfix}]",
            )
            # Train loop
            for x in pbar:
                total_loss += model.distributed_train_step(x)
                num_batches += 1
                pbar.set_description(f"Train Loss: {total_loss / num_batches:.4f}")
            # Test loop
            for x in test:
                total_loss += model.distributed_test_step(x)
                num_batches += 1
            logger.info(f"Test Loss: {total_loss / num_batches:.4f}")

            # Save model weights
            with dist_strategy.scope():
                model.save_weights(save_weights_path)

    logger.debug(f"Model training completed.")

    with dist_strategy.scope():  # not sure if the scope is needed
        model.save_weights(save_weights_path)

    # Save embeddings
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    embeddings_folder = os.path.join(
        "embeddings", f"{args.model}{f'-{args.type}' if args.type else ''}"
    )
    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    domain_embeddings = (
        model.domain_embeddings.embeddings.numpy()
    )  # TODO may break if model class uses a different variable name; use a get_embeddings() function instead
    np.save(
        os.path.join(
            embeddings_folder, f"embeddings-{os.path.splitext(checkpoint_name)[0]}.npy"
        ),
        domain_embeddings,
    )

    logger.debug("Starting model evaluation...")
    if not args.distribute:
        model.evaluate(x=test, y=None, batch_size=args.bs)
    else:  # TODO evaluate can't be used with DistributedDataset, have to loop manually
        pass
    logger.debug("Model evaluation completed.")


if __name__ == "__main__":
    main()
