const NArray = require('./NArray');

/**
 * 
 * @param {} vec (m,n)*(n,p) = (m,p)
 */
function mtxDot(mn, np) {
  let mnShape = mn.shape;
  let npShape = np.shape;
  if (mnShape[1] !== npShape[0] || mnShape.length != 2 || npShape.length != 2) {
    throw new Error('mtxDot:input error: ' + mnShape + '*' + npShape);
  }
  let [m, n] = mn.shape;
  let p = np.shape[1];
  let rt = new NArray([m, p]);

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      sum = 0;
      for (let k = 0; k < n; k++) {
        sum += mn.get([i, k]) * np.get([k, j]);
      }
      rt.set([i, j], sum);
    }
  }
  return rt;
};

module.exports = {
  mtxDot
};


if (require.main === module) {
  //done
  // let t0 = new Date().getTime();
  // var a = new NArray([28 * 28, 32]);

  // for (let i = 0; i < 28 * 28; i++) {
  //   for (let j = 0; j < 32; j++) {
  //     a.set([i, j], Math.random());
  //   }
  // }
  // var b = new NArray([512, 28 * 28]);
  // for (let i = 0; i < 512; i++) {
  //   for (let j = 0; j < 28 * 28; j++) {
  //     b.set([i, j], Math.random());
  //   }
  // }
  // let t1 = new Date().getTime();
  // console.log('init time:', t1 - t0);
  // var c = mtxDot(b, a);
  // let t2 = new Date().getTime();
  // console.log('init time:', t2 - t1);

  //done
  var a = new NArray([3, 2]);
  let k = 1;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 2; j++) {
      a.set([i, j], k++);
    }
  }
  console.log(a.toString());
  var b = new NArray([2, 4]);
  for (let i = 0; i < b.numEle; i++) {
    b.array[i] = i;
  }
  console.log(b.toString());
  console.log('（3,2）*（2,4）--------------------------------');
  console.log(mtxDot(a, b).toString());
  console.log('b--------------------------------');
  b = new NArray([2, 6, 4, 3]);
  for (let i = 0; i < b.numEle; i++) {
    b.array[i] = i;
  }
  console.log(b.toString());
  console.log('b[1,:]--------------------------------');
  var c = b.get([1,]);
  console.log(c.toString());

  console.log('b[1, 0:3, 0:3]--------------------------------');
  c = b.get([1, [0, 3], [0, 3]]);
  console.log(c.toString());
}