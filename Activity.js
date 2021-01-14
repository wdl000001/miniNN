const NArray = require('./NArray');
const Matrix = require('./Matrix');
const Layer = require('./BaseLayer');

const BaseLayer = Layer.BaseLayer;

function relu() {
  BaseLayer.call(this);
  this.type = 'relu';
  this.func = function(x) {
    let array = x.array;
    let out = new NArray(x.shape);
    let outArr = out.array;
    for (let i = 0; i < array.length; i++) {
      const element = array[i];
      outArr[i] = element > 0 ? element : 0;
    }
    this.input_shape = x.shape;
    this.output_shape = out.shape;
    this.input_values = x;
    this.output_values = out;
    return out;
  };
  this.d_func = function(gradient_next) {
    let d = this.output_values.mapElements( el => el > 0 ? el : 0);
    let gradient = d.mul(gradient_next, d);
    return gradient;
  };
}

function sigmoid() {
  BaseLayer.call(this);
  this.type = 'sigmoid';
  this.func = function(x) {
    let out = x.mapElements(el => 1 / (1 + Math.exp(el)));
    this.input_shape = x.shape;
    this.output_shape = out.shape;
    this.input_values = x;
    this.output_values = out;
    return out;
  };
  this.d_func = function(gradient_next) {
    let d = this.output_values.mapElements(el => el * (1 - el));
    let gradient = d.mul(gradient_next, d);
    return gradient;
  };
}

function softmax() {
  BaseLayer.call(this);
  this.type = 'softmax';
  this.func = function(x) {
    //x.shape: (m, dim)
    let [m, dim] = x.shape;
    let ex = new NArray(x.shape);
    for (let i = 0; i < m; i++) {
      let max = -Number.MAX_VALUE;
      for (let j = 0; j < dim; j++) {
        let v = x.get([i, j]);
        if (v > max) {
          max = v;
        }
      }
      for (let j = 0; j < dim; j++) {
        ex.set([i, j], x.get([i, j]) - max);
      }
    }
    ex = ex.mapElements((xi) => Math.exp(xi), ex);
    // let ex = x.mapElements((xi) => Math.exp(xi));
    let sum = new NArray([dim, 1]);
    sum.array.fill(1);
    sum = Matrix.mtxDot(ex, sum);// m, 1
    let out = new NArray(x.shape);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < dim; j++) {
        out.set([i, j], ex.get([i, j]) / (sum.get([i]) + 1e-8));
      }
    }
    this.ex = ex;
    this.sum = sum;
    this.input_shape = x.shape;
    this.output_shape = out.shape;
    return out;
  };
  this.d_func = function(gradient_next) {
    let [m, dim] = this.output_shape;
    let d = new NArray([m, dim, dim]);
    for (let k = 0; k < m; k++) {
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          // daj/dai
          let gradient = 0;
          let exj = this.ex.get([k, j]);
          let exi = this.ex.get([k, i]);
          let sum = this.sum.get([k]);
          if (i == j) {
            gradient = (sum - exi) * exi / (sum * sum + 1e-8);
          } else {
            gradient = -exj * exi / (sum * sum + 1e-8);
          }
          d.set([k, j, i], gradient);
        }
      }
    }
    ////////////////////////////////
    let gradient = new NArray(this.input_shape); //[m, dim]
    for (let j = 0; j < m; j++) {
      let dj = d.get([j]);
      let gj = gradient_next.get([j], true);
      let g = Matrix.mtxDot(gj, dj); // [1, dim] * [dim, dim] = [1,dim]
      gradient.set([j], g);
    }
    return gradient;
  };
}

module.exports = {
  softmax,
  relu,
  sigmoid
};