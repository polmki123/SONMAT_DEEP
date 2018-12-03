var fs = require('fs');
var data = fs.readFileSync(__dirname + '/1001my.png');
var PNG = require('pngjs').PNG;
var png = PNG.sync.read(data);

console.log("feel not good ");
console.log(png)