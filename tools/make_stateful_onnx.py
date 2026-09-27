"""Build dfn3/models/dfn3_stateful.onnx from the official DeepFilterNet3 ONNX export.

The official export (DeepFilterNet3_onnx.tar.gz: enc.onnx, erb_dec.onnx, df_dec.onnx) is meant
for whole-sequence inference: every GRU starts from h0 = 0 and every causal time-convolution
zero-pads its first frames. Running it chunk-by-chunk therefore restarts the recurrent state at
each chunk, and the DFN3 GRUs keep the difference for a long time (measured: the mask differs by
~0.06 on average even 12 s after a cold start). To process long files in bounded memory and still
get *the same* output as one pass over the whole file, this script:

  1. replaces the three causal time-padding `Pad` nodes (feat_erb: 2 frames, feat_spec: 2 frames,
     c0 in df_dec: 4 frames) by `Concat(ctx_in, x)` and exports the last 2/2/4 frames as ctx_out;
  2. feeds every GRU's initial_h from a new input and exports its final state (Y_h);
  3. merges enc + erb_dec + df_dec into ONE graph (intermediate tensors stay inside the runtime,
     one `session.run` per chunk);
  4. drops the unused outputs (lsnr, DF alpha) and prunes dead nodes.

Weights are copied bit-for-bit; no retraining, no re-export from PyTorch.

inputs : feat_erb [1,1,S,32], feat_spec [1,2,S,96]  (already shifted by conv_lookahead = 2)
         erb_ctx [1,1,2,32], spec_ctx [1,2,2,96], c0_ctx [1,64,4,96],
         h_enc [1,1,256], h_erb0 [1,1,256], h_erb1 [1,1,256], h_df0 [1,1,256], h_df1 [1,1,256]
outputs: m [1,1,S,32] (ERB gains), coefs [1,S,96,10] (DF coefs, order 5 x re/im),
         <state>_out for every state input (feed back into the next chunk; zeros for the first)

Usage: python make_stateful_onnx.py <dir with enc.onnx/erb_dec.onnx/df_dec.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HID = 256


def find(g, name):
    for i, nd in enumerate(g.node):
        if nd.name == name:
            return i, nd
    raise KeyError(name)


def add_io(coll, name, shape):
    coll.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))


def pad_to_concat(g, pad_name, ctx, nframes, ch, freq):
    """Pad(x, time-front=nframes) -> Concat(ctx_in, x); ctx_out = last nframes frames."""
    i, nd = find(g, pad_name)
    x, y = nd.input[0], nd.output[0]
    g.node.remove(nd)
    g.node.insert(i, helper.make_node("Concat", [ctx, x], [y], axis=2, name=pad_name + "_ctx"))
    add_io(g.input, ctx, [1, ch, nframes, freq])
    for nm, v in (("starts", [-nframes]), ("ends", [np.iinfo(np.int64).max]), ("axes", [2])):
        g.initializer.append(numpy_helper.from_array(np.array(v, np.int64), f"{ctx}_{nm}"))
    g.node.insert(i + 1, helper.make_node(
        "Slice", [y, f"{ctx}_starts", f"{ctx}_ends", f"{ctx}_axes"], [ctx + "_out"], name=pad_name + "_ctx_out"))
    add_io(g.output, ctx + "_out", [1, ch, nframes, freq])


def gru_state(g, gru_name, h):
    i, nd = find(g, gru_name)
    assert nd.op_type == "GRU" and len(nd.input) == 6, nd
    attrs = {a.name: helper.get_attribute_value(a) for a in nd.attribute}
    assert attrs.get("hidden_size") == HID and attrs.get("direction", b"forward") in (b"forward", "forward")
    nd.input[5] = h
    add_io(g.input, h, [1, 1, HID])
    g.node.insert(i + 1, helper.make_node("Identity", [nd.output[1]], [h + "_out"], name=gru_name + "_h_out"))
    add_io(g.output, h + "_out", [1, 1, HID])


def drop_outputs(g, names):
    for o in [o for o in g.output if o.name in names]:
        g.output.remove(o)


def prefix(g, pfx, keep):
    ren = lambda n: n if (n == "" or n in keep) else pfx + n  # noqa: E731
    for nd in g.node:
        nd.name = pfx + nd.name
        for k, v in enumerate(nd.input):
            nd.input[k] = ren(v)
        for k, v in enumerate(nd.output):
            nd.output[k] = ren(v)
    for t in list(g.initializer) + list(g.sparse_initializer):
        t.name = ren(t.name)
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        vi.name = ren(vi.name)


def prune(g):
    needed = {o.name for o in g.output}
    keep = []
    for nd in reversed(g.node):
        if any(o in needed for o in nd.output):
            keep.append(nd)
            needed.update(x for x in nd.input if x)
    keep.reverse()
    removed = len(g.node) - len(keep)
    del g.node[:]
    g.node.extend(keep)
    inits = [t for t in g.initializer if t.name in needed]
    del g.initializer[:]
    g.initializer.extend(inits)
    vis = [v for v in g.value_info if v.name in needed]
    del g.value_info[:]
    g.value_info.extend(vis)
    return removed


def main():
    src, dst = sys.argv[1], sys.argv[2]
    enc = onnx.load(f"{src}/enc.onnx")
    erb = onnx.load(f"{src}/erb_dec.onnx")
    dfd = onnx.load(f"{src}/df_dec.onnx")

    ge, gr, gd = enc.graph, erb.graph, dfd.graph
    pad_to_concat(ge, "/erb_conv0/0/Pad", "erb_ctx", 2, 1, 32)
    pad_to_concat(ge, "/df_conv0/0/Pad", "spec_ctx", 2, 2, 96)
    gru_state(ge, "/emb_gru/GRU", "h_enc")
    gru_state(gr, "/emb_gru/GRU", "h_erb0")
    gru_state(gr, "/emb_gru/GRU_1", "h_erb1")
    gru_state(gd, "/df_gru/gru/GRU", "h_df0")
    gru_state(gd, "/df_gru/gru/GRU_1", "h_df1")
    pad_to_concat(gd, "/df_convp/df_convp.0/Pad", "c0_ctx", 4, 64, 96)

    shared = {"e0", "e1", "e2", "e3", "emb", "c0"}
    io = lambda g: {v.name for v in list(g.input) + list(g.output)}  # noqa: E731
    prefix(ge, "enc", io(ge))
    prefix(gr, "erb", io(gr) | shared)
    prefix(gd, "dfd", io(gd) | shared)
    # the DF decoder's 2nd output is the (unused) DF alpha; rename then drop it
    drop_outputs(gd, {o.name for o in gd.output if o.name not in ("coefs", "h_df0_out", "h_df1_out", "c0_ctx_out")})

    g = helper.make_graph(
        nodes=list(ge.node) + list(gr.node) + list(gd.node),
        name="dfn3_stateful",
        inputs=[v for v in ge.input] + [v for v in gr.input if v.name not in shared]
        + [v for v in gd.input if v.name not in shared],
        outputs=[o for o in gr.output if o.name == "m"] + [o for o in gd.output if o.name == "coefs"]
        + [o for o in list(ge.output) + list(gr.output) + list(gd.output) if o.name.endswith("_out")],
        initializer=list(ge.initializer) + list(gr.initializer) + list(gd.initializer),
        value_info=list(ge.value_info) + list(gr.value_info) + list(gd.value_info),
    )
    removed = prune(g)
    # producer_name は初版のまま。変えると出力のバイト列が変わり、tools/README.md の sha256 と合わなくなる
    model = helper.make_model(g, opset_imports=list(enc.opset_import), producer_name="dfn3web/make_stateful_onnx.py")
    model.ir_version = enc.ir_version
    model.doc_string = ("DeepFilterNet3 (Rikorose/DeepFilterNet, MIT OR Apache-2.0) official ONNX export "
                        "merged into one stateful graph for chunked offline inference.")
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, dst)
    print(f"saved {dst}: {len(g.node)} nodes ({removed} dead removed), "
          f"inputs {[i.name for i in g.input]}, outputs {[o.name for o in g.output]}")


if __name__ == "__main__":
    main()
