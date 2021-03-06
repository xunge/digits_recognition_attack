'use strict';
var imgSrc = '';
var newData = [];
var inputs = [];

$("#img_input").on("change", function (e) {
    var file = e.target.files[0]; //获取图片资源
    // 只选择图片文件
    if (!file.type.match('image.*')) {
        return false;
    }
    var reader = new FileReader();
    reader.readAsDataURL(file); // 读取文件
    // 渲染文件
    reader.onload = function (arg) {
        imgSrc = arg.target.result;
        var img = '<img id="update" src="' + arg.target.result + '" alt="preview"/>';
        $(".preview_box").empty().append(img);
    }
});

var _createClass = function ()
{ function defineProperties(target, props)
{ for (var i = 0; i < props.length; i++)
{ var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false;
descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true;
Object.defineProperty(target, descriptor.key, descriptor); } }
return function (Constructor, protoProps, staticProps)
{ if (protoProps) defineProperties(Constructor.prototype, protoProps);
if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

function _classCallCheck(instance, Constructor)
{ if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }


/* global $ */
var Main = function () {
    function Main() {
        _classCallCheck(this, Main);

        this.canvas = document.getElementById('main');
        this.input = document.getElementById('input');
        this.canvas.width = 449; // 16 * 28 + 1
        this.canvas.height = 449; // 16 * 28 + 1
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.initialize();
    }

    _createClass(Main, [{
        key: 'initialize',
        value: function initialize() {
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.fillRect(0, 0, 449, 449);
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(0, 0, 449, 449);
            this.ctx.lineWidth = 0.05;
            $('#output').find('td').text('').removeClass('success');
            $('#origin').attr('src', '')
            $('#fgsm_mnist').attr('src', '')
            $('#pgd_mnist').attr('src', '')
        }
    }, {
        key: 'onMouseDown',
        value: function onMouseDown(e) {
            this.canvas.style.cursor = 'default';
            this.drawing = true;
            this.prev = this.getPosition(e.clientX, e.clientY);
        }
    }, {
        key: 'onMouseUp',
        value: function onMouseUp() {
            this.drawing = false;
        }
    }, {
        key: 'onMouseMove',
        value: function onMouseMove(e) {
            if (this.drawing) {
                var curr = this.getPosition(e.clientX, e.clientY);
                this.ctx.lineWidth = 25;
                this.ctx.lineCap = 'round';
                this.ctx.beginPath();
                this.ctx.moveTo(this.prev.x, this.prev.y);
                this.ctx.lineTo(curr.x, curr.y);
                this.ctx.stroke();
                this.ctx.closePath();
                this.prev = curr;
            }
        }
    }, {
        key: 'getPosition',
        value: function getPosition(clientX, clientY) {
            var rect = this.canvas.getBoundingClientRect();
            return {
                x: clientX - rect.left,
                y: clientY - rect.top
            };
        }
    }, {
        key: 'drawInput',
        value: function drawInput() {
            var img = new Image();
            img.onload = function () {
                inputs = [];
                var small = document.createElement('canvas').getContext('2d');
                small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
                var data = small.getImageData(0, 0, 28, 28).data;
                for (var i = 0; i < 28; i++) {
                    for (var j = 0; j < 28; j++) {
                        var n = 4 * (i * 28 + j);
                        inputs[i * 28 + j] = (data[n] + data[n + 1] + data[n + 2]) / 3;
                    }
                }
                if (Math.min.apply(Math, inputs) === 255) {
                    return;
                }
                $.ajax({
                    url: "/drawInput",
                    data: {"inputs": JSON.stringify(inputs)},
                    type: "POST"
                });
                var src = "/static/img/origin.png"+"?t=" + Math.random();
                var timeout=setTimeout(function () {
                    $("#origin").attr('src', src)
                }, 100);
            };
            img.src = this.canvas.toDataURL();
        }
    }, {
        key: 'attack_fgsm',
        value: function attack_fgsm() {
            newData = [];
            $.ajax({
                url: "/attack_fgsm",
                data: {"inputs": JSON.stringify(inputs)},
                type: "POST"
            });
            var src = "/static/img/fgsm_mnist.png"+"?t=" + Math.random();
            var timeout=setTimeout(function () {
                $("#fgsm_mnist").attr('src', src)
            }, 100);
        }
    }, {
        key: 'attack_pgd',
        value: function attack_pgd() {
            newData = [];
            $.ajax({
                url: "/attack_pgd",
                data: {"inputs": JSON.stringify(inputs)},
                type: "POST"
            });
            var src = "/static/img/pgd_mnist.png"+"?t=" + Math.random();
            var timeout=setTimeout(function () {
                $("#pgd_mnist").attr('src', src)
            }, 200);
        }
    }, {
        key: 'recognizeDraw',
        value: function recognizeDraw() {
            $.ajax({
                url: "/process",
                data: {"inputs": JSON.stringify(inputs)},
                type: "POST",
                async: false,
                dataType: "json",
                success: function (data) {
                    var newData = eval(data);   //#将字符串转换为整数。
                    for (var _i = 0; _i < 3; _i++) {
                        var max = 0;
                        var max_index = 0;
                        for (var _j = 0; _j < 10; _j++) {
                            var value = Math.round(newData[_i][_j] * 1000);
                            if (value > max) {
                                max = value;
                                max_index = _j;
                            }
                            var digits = String(value).length;
                            for (var k = 0; k < 3 - digits; k++) {
                                value = '0' + value;
                            }
                            var text = '0.' + value;
                            if (value > 999) {
                                text = '1.000';
                            }
                            $('#output').find('tr').eq(_j + 1).find('td').eq(_i).text(text);
                        }
                        for (var _j2 = 0; _j2 < 10; _j2++) {
                            if (_j2 === max_index) {
                                $('#output').find('tr').eq(_j2 + 1).find('td').eq(_i).addClass('success');
                            } else {
                                $('#output').find('tr').eq(_j2 + 1).find('td').eq(_i).removeClass('success');
                            }
                        }
                    }
                }
            });
        }
    }, {
        key: 'updateInput',
        value: function updateInput() {
            var img = new Image();
            img.onload = function () {
                inputs = [];
                var small = document.createElement('canvas').getContext('2d');
                small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28);
                var data = small.getImageData(0, 0, 28, 28).data;
                console.log(data);
                for (var i = 0; i < 28; i++) {
                    for (var j = 0; j < 28; j++) {
                        var n = 4 * (i * 28 + j);
                        inputs[i * 28 + j] = (data[n] + data[n + 1] + data[n + 2]) / 3;
                    }
                }
                console.log(inputs);
                if (Math.min.apply(Math, inputs) === 255) {
                    return;
                }
                newData = [];
                $.ajax({
                    url: "/drawInput",
                    data: {"inputs": JSON.stringify(inputs)},
                    type: "POST"
                });
                var src = "/static/img/origin.png"+"?t=" + Math.random();
                var timeout=setTimeout(function () {
                    $("#origin").attr('src', src)
                }, 100);
            };
            img.src = imgSrc;
        }
    }]);
    return Main;
}();

$(function () {
    var main = new Main();
    $('#clear').click(function () {
        main.initialize();
    });
});

$(function () {
    var main = new Main();
    $('#recognizeDraw').click(function () {
        main.recognizeDraw();
    });
});

$(function () {
    var main = new Main();
    $('#drawInput').click(function () {
        main.drawInput();
    });
});

$(function () {
    var main = new Main();
    $('#updateInput').click(function () {
        main.updateInput();
    });
});

$(function () {
    var main = new Main();
    $('#attack_btn_fgsm').click(function () {
        main.attack_fgsm();
    });
});

$(function () {
    var main = new Main();
    $('#attack_btn_pgd').click(function () {
        main.attack_pgd();
    });
});