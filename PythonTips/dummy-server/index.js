var express = require('express');
var app = express();

app.get('/', function(req, res){
	res.send('Hello World!');
});

app.get('/slow', function(req, res){
	setTimeout(function() {
		res.send('Slow Response');
	}, 1000);
});

app.get('/slower', function(req, res){
	setTimeout(function () {
		res.send('Slower Response');
	}, 3000);
});

app.get('/super-slow', function(req, res) {
	setTimeout(function () {
		res.send('Super Slow Response');
	}, 3000);
});

app.set('port', (process.env.PORT || 3000));
app.listen(app.get('port'), function() {
	console.log('Server started: http://localhost:' + app.get('port') + '/');
});
