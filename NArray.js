/**
 * 
 * @param {} shape 形如 [3, 32,32]的数组
 */
function NArray(shape) {
  let numEle = 1;
  shape.forEach(o => { numEle *= o; });
  this.numEle = numEle;
  this.shape = shape;
  // let buffer = this.buffer = new ArrayBuffer(numEle * 4);//一个float32占4字节
  // this.array = new Float32Array(buffer);
  let array = this.array = new Array(numEle); //直接用Array效率大概是Float32Array效率的2.2~2.5倍
  array.fill(0.0);
}
//todo
NArray.prototype.reshape = function(shape) {
  if (shape.indexOf(-1) != -1) {
    let dim = this.numEle;
    let ix = -1;
    shape.forEach((s, i) => {
      if (s != -1) {
        dim /= s;
      } else if (ix == -1) {
        ix = i;
      } else {
        throw new Error('reshape err');
      }
    });
    shape[ix] = dim;
  }
  this.shape = shape;
};
/**
 * 切片操作
 * @param {} subShape array，长度小于等于shape，每个索引可以是正数，array,或者null,用null表示该维度所有数据。(暂不支持负数索引),
 * 返回的数据是原数据的copy，要修改原数据需要用set
 * @eg narr.get([3,5,null]); narr.get([[0,2], null,5])
 */
NArray.prototype.get = function(subShape, keepDim = false) {
  let { shape } = this;
  let { startIx, lenIx } = calcSubShape(subShape, shape);
  //塞数据
  let rt = new NArray(lenIx);
  let rtArgs = rt.numEle;
  let subArr = rt.array;
  let fullArr = this.array;
  for (let i = 0; i < rtArgs; i++) {
    let axis = ix2Axis(i, lenIx);
    offsetAxis(axis, startIx);
    let k = axis2Ix(axis, shape);
    subArr[i] = fullArr[k];
  }
  if (keepDim) {
    return rt;
  }
  //压缩轴
  let rtShape = [];
  lenIx.forEach(s => {
    if (s > 1) {
      rtShape.push(s);
    }
  });
  rt.shape = rtShape;
  if (rtShape.length === 0) {
    return rt.array[0];
  }
  return rt;
};

NArray.prototype.set = function(subShape, value) {
  let { shape } = this;
  let { startIx, lenIx } = calcSubShape(subShape, shape);
  //塞数据
  let numEle = 1;
  lenIx.forEach(s => { numEle *= s; });
  if (typeof value === 'number') {
    let fullArr = this.array;
    for (let i = 0; i < numEle; i++) {
      let axis = ix2Axis(i, lenIx);
      offsetAxis(axis, startIx);
      let k = axis2Ix(axis, shape);
      fullArr[k] = value;
    }
  } else if (value.numEle === numEle) {
    let subArr = value.array;
    let fullArr = this.array;
    for (let i = 0; i < numEle; i++) {
      let axis = ix2Axis(i, lenIx);
      offsetAxis(axis, startIx);
      let k = axis2Ix(axis, shape);
      fullArr[k] = subArr[i];
    }
  } else {
    throw new Error(`shape of value error,target shape is ${lenIx} value shape is ${value.shape} `);
  }
};

NArray.prototype.siteBySite = function(b, func, out, create = true) {
  if (!out) {
    if (create) {
      out = new NArray(this.shape);
    } else {
      out = this;
    }
  }
  let arr = this.array;
  let brr = b.array;
  let orr = out.array;
  for (let i = 0; i < arr.length; i++) {
    orr[i] = func(arr[i], brr[i]);
  }
  return out;
};
NArray.prototype.add = function(b, out, create = true) {
  return this.siteBySite(b, (e1, e2) => e1 + e2, out, create);
};

NArray.prototype.sub = function(b, out, create = true) {
  return this.siteBySite(b, (e1, e2) => e1 - e2, out, create);
};

NArray.prototype.mul = function(b, out, create = true) {
  return this.siteBySite(b, (e1, e2) => e1 * e2, out, create);
};

NArray.prototype.div = function(b, out, create = true) {
  return this.siteBySite(b, (e1, e2) => {
    if (e2 >= 0 && e2 < 1e-8) {
      return e1 / 1e-8;
    } else if (e2 < 0 && e2 > -1e-8) {
      return -e1 / 1e-8;
    } else {
      return e1 / e2;
    }
  }, out, create);
};

NArray.prototype.scale = function(b, out, create = false) {
  if (!out) {
    if (create) {
      out = new NArray(this.shape);
    } else {
      out = this;
    }
  }
  let arr = this.array;
  let orr = out.array;
  for (let i = 0; i < arr.length; i++) {
    orr[i] = arr[i] * b;
  }
  return out;
};

NArray.prototype.mapElements = function(func, out, create = true) {
  if (!out) {
    if (create) {
      out = new NArray(this.shape);
    } else {
      out = this;
    }
  }
  let arr = this.array;
  let orr = out.array;
  for (let i = 0; i < arr.length; i++) {
    orr[i] = func(arr[i]);
  }
  return out;
};

NArray.prototype.transpose = function(create = true) {
  let out = this;
  let shape = this.shape;
  let newShape = shape.concat().reverse();
  if (create) {
    out = new NArray(newShape);
  } else {
    out.shape = newShape;
  }
  let origin = this.array.concat();
  let numEle = this.numEle;
  let arr = out.array;
  for (let i = 0; i < numEle; i++) {
    let axis = ix2Axis(i, shape);
    axis.reverse();
    let newIx = axis2Ix(axis, newShape);
    arr[newIx] = origin[i];
  }
  return out;
};

NArray.prototype.toString = function() {
  let times = [1];
  let { shape, array, numEle } = this;
  shape = shape.concat();
  shape.reverse();
  for (let i = 1; i < shape.length; i++) {
    times[i] = times[i - 1] * shape[i - 1];
  }
  let s = '';
  for (let i = 0; i < numEle; i++) {
    s += array[i];
    let n = 0;
    for (let j = 1; j < times.length; j++) {
      if (i != 0 && (i + 1) % times[j] === 0) {
        n++;
      }
    }
    if (n > 0 && i !== numEle - 1) {
      s += ' ';
      for (let k = 0; k < n; k++) { s += '|'; }
      s += ' ';
    }
    if (n == 0) { s += ', '; }
    else if (n > 1 || (shape.length === 2 && n === 1)) { s += '\n'; }
  }
  return s;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
function ix2Axis(ix, shape) {
  let axis = [];
  //shape: [i1,i2,i3,i4...]
  let times = new Array(shape.length);
  times[times.length - 1] = 1;
  for (let i = times.length - 2; i >= 0; i--) {
    times[i] = times[i + 1] * shape[i + 1];
  }
  for (let i = 0; i < times.length; i++) {
    let a = Math.floor(ix / times[i]);
    axis.push(a);
    ix = ix % times[i];
  }
  return axis;
}

function axis2Ix(axis, shape) {
  let times = new Array(shape.length);
  times[times.length - 1] = 1;
  for (let i = times.length - 2; i >= 0; i--) {
    times[i] = times[i + 1] * shape[i + 1];
  }
  let ix = 0;
  for (let i = 0; i < axis.length; i++) {
    ix += axis[i] * times[i];
  }
  return ix;
}

function offsetAxis(axis, offsetAxis) {
  for (let i = 0; i < axis.length; i++) {
    axis[i] += offsetAxis[i];
  }
}

function calcSubShape(subShape, shape) {
  //推断return shape
  let startIx = [];
  let lenIx = [];
  // let rtShape = []; => lenIx
  for (let i = 0; i < shape.length; ++i) {
    if (i >= subShape.length) {
      subShape[i] = null;
    }
    let ix = subShape[i];
    if (typeof ix === 'number') {
      startIx[i] = ix;
      lenIx[i] = 1;
      ix = subShape[i] = [ix, ix + 1];
    } else if (ix == null) {
      let dimLen = shape[i];
      ix = subShape[i] = [0, dimLen];
      startIx[i] = 0;
      lenIx[i] = dimLen;
    } else if (Array.isArray(ix)) {
      startIx[i] = ix[0];
      lenIx[i] = ix[1] - ix[0];
    }
  }
  return { startIx, lenIx };
}


module.exports = NArray;

if (require.main === module) {
  // done
  let axis = [2, 5];
  console.log(axis);
  let ix = axis2Ix([0, 1], axis);
  console.log([0, 1], ix);
  r_axis = ix2Axis(ix, axis);
  console.log(r_axis);

  ix = axis2Ix([0, 3], axis);
  console.log([0, 3], ix);
  r_axis = ix2Axis(ix, axis);
  console.log(r_axis);

  ix = axis2Ix([1, 1], axis);
  console.log([1, 1], ix);
  r_axis = ix2Axis(ix, axis);
  console.log(r_axis);

  ix = axis2Ix([1, 3], axis);
  console.log([1, 3], ix);
  r_axis = ix2Axis(ix, axis);
  console.log(r_axis);

  console.log('---', axis);
  //done
  var arr = new NArray(axis);
  // console.log(arr.shape);
  arr.set([0, 1], 1);
  arr.set([0, 3], 3);
  arr.set([1, 1], 6);
  arr.set([1, 3], 8);
  console.log(arr.toString());
  console.log(arr.array);
}