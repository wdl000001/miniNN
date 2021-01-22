const NArray = require('./NArray');
const Matrix = require('./Matrix');

function BaseLayer() {
  this.weights = null;
  this.biases = null;
  this.func = null;
  this.d_func = null;
  this.init = null;
  this.isLayer = true;
  this.l2 = 0;
  this.learn_rate = 0.001;
  this.type = 'BaseLayer';
}

function NN(inputDim, outputDim) {
  BaseLayer.call(this);
  this.type = 'NN';
  this.weights = new NArray([inputDim, outputDim]);
  this.biases = new NArray([outputDim]);
  this.init = function() {
    let arr = this.weights.array;
    for (let i = 0; i < arr.length; i++) {
      arr[i] = Math.random() - 0.5;
    }
  };
  this.func = function(x) {
    // X*W=Y : (m, in) * (in, out) => (m, out)
    let out = Matrix.mtxDot(x, this.weights);
    addBiases(out, this.biases);
    this.input_shape = x.shape;
    this.output_shape = out.shape;
    this.input_values = x;
    this.output_values = out;
    return out;
  };
  this.d_func = function(gradient_next) {
    let d = this.weights.transpose(true);// (out, in)
    //梯度回传
    let gradient = Matrix.mtxDot(gradient_next, d);
    //更新参数
    let m = gradient_next.shape[0];
    let dw = Matrix.mtxDot(this.input_values.transpose(), gradient_next);
    dw.scale(1 / m, dw, false);
    let wArr = this.weights.array;
    let dwArr = dw.array;
    for (let j = 0; j < wArr.length; j++) {
      wArr[j] -= this.learn_rate * (dwArr[j] + this.l2 * wArr[j]);
    }
    let bArr = this.biases.array;
    let sum = new NArray([1, m]);
    sum.array.fill(1.0)
    sum = Matrix.mtxDot(sum, gradient_next);
    sum.scale(1 / m, sum);
    let gradientArr = sum.array;
    for (let j = 0; j < bArr.length; j++) {
      bArr[j] -= this.learn_rate * gradientArr[j];
    }
    return gradient;
  };
}

function Flatten() {
  BaseLayer.call(this);
  this.type = 'Flatten';

  this.init = function() {
  };
  this.func = function(x) {
    let shape = x.shape;
    let k = shape[0];
    let oth = 1;
    for (let i = 1; i < shape.length; ++i) {
      oth *= shape[i];
    }
    let out = new NArray([k, oth]);
    out.array = x.array.concat();
    this.input_shape = x.shape;
    this.output_shape = out.shape;
    this.input_values = x;
    this.output_values = out;
    return out;
  };
  this.d_func = function(gradient_next) {
    gradient_next.reshape(this.input_shape);
    return gradient_next;
  };
}

function Conv2d(w, h, inputDim, outputDim, slide_w = 1, slide_h = 1) {
  BaseLayer.call(this);
  this.type = 'Conv2d';
  this.weights = new NArray([w * h * inputDim, outputDim]);
  this.kw = w;
  this.kh = h;
  this.sw = slide_w;
  this.sh = slide_h;
  this.biases = new NArray([outputDim]);
  this.init = function() {
    let arr = this.weights.array;
    for (let i = 0; i < arr.length; i++) {
      arr[i] = Math.random() - 0.5;
    }
  };
  this.func = function(x) {
    // X*W=Y : (m, in) * (in, out) => (m, out)
    let [m, w, h, inDim] = x.shape;
    let { kw, kh, sw, sh } = this;
    var pw = w + kw - 1;
    var ph = w + kh - 1;
    var pl = (pw - w) % 2 == 0 ? (pw - w) / 2 : (pw - w - 1) / 2;
    var pt = (ph - h) % 2 == 0 ? (ph - h) / 2 : (ph - h - 1) / 2;
    this.pl = pl;
    this.pt = pt;
    this.pw = pw;
    this.ph = ph;
    let paddingX = new NArray([m, pw, ph, inDim]);
    paddingX.set([null, [pl, pl + w], [pt, pt + h], null], x);
    let out_w = Math.ceil(w / sw);
    let out_h = Math.ceil(h / sh);
    let inputDim = kw * kh * inDim;
    let patches = new NArray([m, out_w, out_h, inputDim]);
    for (let k = 0; k < m; k++) {
      for (let i = 0; i < out_w; i++) {
        for (let j = 0; j < out_h; j++) {
          let patch = paddingX.get([k, [i * sw, i * sw + kw], [j * sh, j * sh + kh]]); //[kw,kh,inDim]
          patch.reshape([inputDim]);
          patches.set([k, i, j], patch);
        }
      }
    }
    patches.reshape([m * out_w * out_h, inputDim]);
    this.patches = patches;

    let out = Matrix.mtxDot(patches, this.weights);
    addBiases(out, this.biases);
    out.reshape([m, out_w, out_h, this.weights.shape[1]]);
    this.input_shape = x.shape;
    this.output_shape = out.shape;
    return out;
  };
  this.d_func = function(gradient_next) {
    let d = this.weights.transpose(true);// (out, in)
    gradient_next.reshape([-1, d.shape[0]]);
    let gradient = Matrix.mtxDot(gradient_next, d); // [outDim, inKW*inKH*inDim]
    //conv2d梯度反传
    gradient = this.back(gradient);
    //TODO
    let m = gradient_next.shape[0];
    let dw = Matrix.mtxDot(this.patches.transpose(), gradient_next);// [indim, m] * [m, output]
    dw.scale(1 / m, dw, false);
    let wArr = this.weights.array;
    let dwArr = dw.array;
    for (let j = 0; j < wArr.length; j++) {
      wArr[j] -= this.learn_rate * (dwArr[j] + this.l2 * wArr[j]);
    }
    let bArr = this.biases.array;
    let sum = new NArray([1, m]);
    sum.array.fill(1.0)
    sum = Matrix.mtxDot(sum, gradient_next);
    sum.scale(1 / m, sum);
    let gradientArr = sum.array;
    for (let j = 0; j < bArr.length; j++) {
      bArr[j] -= this.learn_rate * gradientArr[j];
    }
    return gradient;
  };
  this.back = function(gradient) {
    let [m, w, h, inDim] = this.input_shape;
    let [_, out_w, out_h, _1] = this.output_shape;
    let { kw, kh, sw, sh, pl, pt, pw, ph } = this;
    let out = new NArray([m, pw, ph, inDim]);

    gradient.reshape([m, out_w, out_h, -1]);

    for (let k = 0; k < m; k++) {
      for (let i = 0; i < out_w; i++) {
        for (let j = 0; j < out_h; j++) {
          let rowgradient = gradient.get([k, i, j]);//144
          rowgradient.reshape([kw, kh, inDim]);
          out.get([k, [i * sw, i * sw + kw], [j * sh, j * sh + kh]]);
          rowgradient.add(out, rowgradient);
          out.set([k, [i * sw, i * sw + kw], [j * sh, j * sh + kh]], rowgradient);
        }
      }
    }
    out = out.get([null, [pl, pl + w], [pt, pt + h]]);
    return out;
  };
}

function addBiases(input, biases) {
  // input (m, out)
  //biases (1, out)
  let arr = input.array;
  let brr = biases.array;
  let [m, n] = input.shape;
  let numArgs = input.numArgs;
  for (let i = 0; i < numArgs; i += n) {
    for (let j = 0; j < n; j++) {
      arr[i] += brr[j];
    }
  }
}

module.exports = {
  BaseLayer,
  NN,
  Conv2d,
  Flatten
};