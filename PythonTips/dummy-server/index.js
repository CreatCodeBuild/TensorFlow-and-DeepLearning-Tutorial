var express = require('express');
var app = express();

app.get('/', function(req, res){
	console.log('/');
	res.end('Hello World!');
});

app.get('/slow', function(req, res){
	console.log('/slow');
	setTimeout(function() {
		res.end('Slow Response');
	}, 1000);
});

app.get('/slower', function(req, res){
	console.log('/slower');
	setTimeout(function () {
		res.end('Slower Response');
	}, 2000);
});

app.get('/super-slow', function(req, res) {
	console.log('/super-slow');
	setTimeout(function () {
		res.end('Super Slow Response');
	}, 3000);
});

app.set('port', (process.env.PORT || 3000));
app.listen(app.get('port'), function() {
	console.log('Server started: http://localhost:' + app.get('port') + '/');
});
