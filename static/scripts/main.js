recording = false

var final_transcript = "";

$('.camfailed').hide();

function checkCamera() {
  navigator.mediaDevices.getUserMedia({ video: true , audio: false})
  .catch((error) => {
    $('.attendence').remove();
    $('.camfailed').show();
  });

  setTimeout(checkCamera, 1000);
}

function drawRects(drawing_ctx, rects, text) {
  drawing_ctx.clearRect(0, 0, 640, 480);

  var color;
  if (text === 'DOULEURS') {
      color = 'red';  // Couleur rouge pour la douleur
  } else {
      color = 'green';  // Couleur verte pour l'absence de douleur
  }

  drawing_ctx.beginPath();
  drawing_ctx.rect(rects[0], rects[1], rects[3], rects[2]);
  drawing_ctx.lineWidth = 2;
  drawing_ctx.strokeStyle = color;  // Utilisez la couleur définie
  drawing_ctx.stroke();

  drawing_ctx.fillStyle = color;
  drawing_ctx.font = 'bold 20px Arial'; // Change la taille et le style de la police
  drawing_ctx.fillText(text, rects[0], rects[1] - 10);
}


$(document).ready(() => {
   checkCamera()

  navigator.mediaDevices.getUserMedia({ video: true , audio: false})
  .then((stream) => {
    $('.authvideo').get(0).srcObject = stream;
    var authvideo = $('.authvideo').get(0);

    // get the canvas where to draw face rects and emotions
    const drawing_canvas = document.getElementById('videoCanvas');
    const drawing_ctx = drawing_canvas.getContext('2d');

    drawing_canvas.width = authvideo.width;
    drawing_canvas.height = authvideo.height;

    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');

    canvas.width = authvideo.width;
    canvas.height = authvideo.height;

    function captureImg() {
      ctx.drawImage(authvideo, 0, 0);
      var dataURL = canvas.toDataURL();

      $.ajax({
        type: "POST",
        url: "/upload_image",
        data: dataURL,
        processData: false,
        contentType: false,
        success: function(response) {
          console.log(response);
          if (response['rects'].length > 0) {
            drawRects(drawing_ctx, response['rects'][0], response['pain']);
          }
        },
        error: function(xhr, status, error) {
          console.error(error);
        }
      });
    }

    setInterval(captureImg, 1000);

  })
  .catch(function (error) {
    console.log("La caméra ne fonctionne pas!");
    console.log(error);
    $('.authentification').remove();
    $('.camfailed').show();
  });
});

$('.camfailed').click(() => {
  location.reload();
});
