import wandb
from utils import *
from torch import nn
from dictionary_corpus import *
import transformers
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
            prog='GPT2-Retrain')

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--pretrained_arch", type=str, default="gpt2")
parser.add_argument("--cuda", action="store_true", default=False)
parser.add_argument("--lr", type=float, default = 6e-4)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--load_path", type=str)
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--sched_start", type=float, default=(1.0/3.0))
parser.add_argument("--log_interval", type=int, default=1000)
parser.add_argument("--eval_stride", type=int, default=64)
parser.add_argument("--debug", action="store_true", default=False)

args = vars(parser.parse_args())

data = Corpus(args["data_path"])

config_kwargs = {"vocab_size": len(data.dictionary.word2idx)}
config = transformers.AutoConfig.from_pretrained(args["pretrained_arch"], **config_kwargs)
model = transformers.AutoModelForCausalLM.from_config(config)


if args["cuda"]:
    model.cuda()


optimizer = AdamW(model.parameters(), lr=args["lr"])
scheduler = LinearLR(optimizer, start_factor=args["sched_start"])

if args["load_path"] is not None:
    checkpoint = torch.load(fn, map_location=torch.device("cuda") if args["cuda"] else torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler = checkpoint["scheduler"]

model.gradient_checkpointing_enable()

train = batchify(data.train, args["batch_size"], args["cuda"])
valid = batchify(data.valid, args["batch_size"], args["cuda"])

if not args["debug"]:
    wandb.init(
        project = "GWikiGPT2",
        config = args)

def eval(model, data, seq_len, stride):
    model.eval()
    losses = []
    with torch.no_grad():
        for step in range((len(data)-seq_len)//stride):
            inp = get_batch(data, step, seq_len, stride=stride)
            target = inp.clone()
            target[:,:seq_len - stride] = -100
            outputs = model(inp, labels=target)
            losses.append(outputs.loss)
        avg_loss = torch.mean(torch.stack(losses))
    return avg_loss.item(), torch.exp(avg_loss).item()
    
num_train_batches = len(train)//args["seq_len"]  
num_valid_batches = (len(valid)-args["seq_len"])//args["eval_stride"]  

if min(num_train_batches, num_valid_batches) < 1:
    print("ERROR: Not enough data to allow for batch size/sequence length combination")
    exit()

print(model)
print(args)
print("num train tokens = {}".format(len(data.train)))
print("num train batches/epoch = {}".format(num_train_batches))
print("num valid tokens = {}".format(len(valid)))
print("num valid batches = {}".format(num_valid_batches))
print("num epochs = {}".format(args["epochs"]))
if args["cuda"]: print("Using CUDA")

total_steps = 0
for epoch in range(args["epochs"]):
    print("--- epoch {}".format(epoch))
    losses = []
    scheduler.step()
    for step in range(num_train_batches):
        model.train()
        batch = get_batch(train, step, args["seq_len"])

        if args["debug"]:
            print("DEBUG: batch dims = {}, step {}".format(batch.size(), step))
            for i in range(len(batch)):
                print("\t".join(data.dictionary.idx2word[idx] for idx in batch[i]))

        loss = model(batch, labels=batch, use_cache=False).loss
        losses.append(loss.repeat(batch.size()[0]).detach()) 

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_steps += 1
        if total_steps % args["log_interval"] == 0:
            model.save_pretrained(args["save_path"] + "checkpoints/")

            Path(args["save_path"] + "ptorch/").mkdir(parents=True, exist_ok=True)
            torch.save({"optimizer":optimizer.state_dict(), 
                        "scheduler":scheduler,
                        "model":model.state_dict()}, 
                        args["save_path"] + "ptorch/state-{}-{}.pt".format(epoch, step))
            v_loss, v_ppl = eval(model, valid, args["seq_len"], stride=args["eval_stride"])
            t_loss = torch.mean(torch.stack(losses))
            t_ppl = torch.exp(t_loss).item()
            print("epoch {} | step {} ({}) | train_ppl: {:.4f} | valid ppl: {:.4f}".format(epoch, step, total_steps, t_ppl, v_ppl))
            if not args["debug"]:
                wandb.log({"epoch":epoch,
                           "step": step,
                           "valid_loss":v_loss,
                           "valid_ppl":v_ppl,
                           "train_loss":t_loss.item(),
                           "train_ppl":t_ppl})



if not args["debug"]: wandb.finish()
