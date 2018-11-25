// const { exec, execSync, spawn } = require('child_process');
// var multer = require('multer');
var fs = require('fs');
var svg2ttf = require('svg2ttf');
var svgicons2svgfont = require('svgicons2svgfont');
var fontStream = new svgicons2svgfont({
  fontName: 'myfont'
});
var ImageTracer = require('../public/javascripts/imagetracer_v1.2.1');
var PNG = require('pngjs').PNG;

var files;
// var option = req.body.options;
files = fs.readdirSync(__dirname + '/../200/');
// var option = req.body.options;
var sources=[];
var fileName=[];
// console.log(root_dir);

for(var i=0; i<files.length; i++) {
      sources[i] = '0x' + files[i].substring(0,4);
      fileName[i] = files[i].substring(0,4);
      //console.log('\u0000'.substring(0,2));
}
for(var i=0; i<files.length; i++) {
    let j = i;
    var data = fs.readFileSync(__dirname + '/../200/'+files[i]);
    var png = PNG.sync.read(data);
    var myImageData = {width:64, height:64, data:png.data};
    // var options = {ltres:option.ltres, strokewidth:option.strokewidth, qtres:option.qtres, pathomit:option.pathomit, blurradius:option.blurradius, blurdelta:option.blurdelta};
    // options.pal = [{r:0,g:0,b:0,a:255},{r:255,g:255,b:255,a:255}];
    // options.linefilter=true;

    // let svgstring = ImageTracer.imagedataToSVG( myImageData, options);
    let svgstring = ImageTracer.imagedataToSVG( myImageData);
    fs.writeFileSync(__dirname + '/../svg/' + fileName[i] + '.svg', svgstring); 
    // fs.writeFile('/../svg/' + fileName[i] + '.svg', svgstring, (err) => {
    //   if (err) throw err;
    //   console.log('The file has been saved!');
    // });
    
}
console.log('error');
fontStream.pipe(fs.createWriteStream( __dirname+ '/../svg_fonts/font_ss.svg'))
  .on('finish',function() {
    var ttf = svg2ttf(fs.readFileSync(__dirname+ '/../svg_fonts/font_ss.svg', 'utf8'), {});
    fs.writeFileSync(__dirname + '/../ttf_fonts/myfont.ttf', new Buffer(ttf.buffer));
  })
  .on('error',function(err) {
    console.log(err);
  });
console.log('error');
for (var i=0; i < sources.length; i++) {
  // Writing glyphs
  let glyph1 = fs.createReadStream(__dirname+ '/../svg/' + fileName[i] + '.svg');
  glyph1.metadata = {
    unicode: [String.fromCharCode((sources[i]).toString(10))],
    name: 'glyph' + sources[i]
  };
  fontStream.write(glyph1);
}
fontStream.end();


