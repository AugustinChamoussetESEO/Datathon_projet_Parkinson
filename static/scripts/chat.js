(function() {
  var fakeNum, messages, uctTimer;

  messages = $(".messages-content");

  fakeNum = 0;

  uctTimer = null;

  window.setDate = function() {
    var d, timestamp;
    timestamp = $("<div>").addClass("timestamp");
    d = new Date();
    timestamp.text(d.getHours() + ":" + (d.getMinutes() < 10 ? '0' : '') + d.getMinutes());
    return timestamp.appendTo($('.message:last'));
  };

  window.updateScrollbar = function() {
    return messages.mCustomScrollbar("update").mCustomScrollbar('scrollTo', 'bottom', {
      scrollInertia: 10,
      timeout: 0
    });
  };

  window.pushMessage = function(msg_text) {
    var typing;
    typing = $("<div>").append("<span>").addClass("message typing");
    typing.appendTo($('.mCSB_container'));
    updateScrollbar();
    return setTimeout(function() {
      var msg;
      typing.remove();
      msg = $("<div>").addClass("message");
      msg.text(msg_text);
      msg.appendTo($('.mCSB_container'));
      setDate();
      updateScrollbar();
      const utterThis = new SpeechSynthesisUtterance(msg_text);
      speechSynthesis.speak(utterThis);
      return fakeNum++;
    }, 1000 + (Math.random() * 10) * 100);
  };

  window.insertMessage = function() {
    var msg, msgText;
    msgText = $(".action-box-input").val();
    if (($.trim(msgText) === "") && (final_transcript === "") ) {
      return false;
    }
    msg = $("<div>").addClass("message");
    if($.trim(msgText) === ""){
      console.log(final_transcript);
      msg.text(final_transcript);
    }
    else{
      msg.text(msgText);
    }
    msg.addClass("personal").appendTo($('.mCSB_container'));
    setDate();
    updateScrollbar();
    $(".action-box-input").val(null);
    $(".message.personal.typing").remove();

    clearTimeout(uctTimer);

    console.log(msgText);

    $.ajax({
        type: "POST",
        url: "/message",
        data: msgText,
        contentType: "text/plain",
        success: function(response) {
            pushMessage(response)
        },
        error: function(error) {
            console.log(error)
        }
    });
  };

  $(window).on('keydown', function(e) {
    if (e.which === 13) {
      insertMessage();
      return false;
    }
  });

  $(window).on('load', function() {
    messages.mCustomScrollbar();

    setTimeout(function() {
        $.ajax({
            type: "POST",
            url: "/message",
            data: "<$>",  //conversation starter token
            contentType: "text/plain",
            success: function(response) {
                pushMessage(response)
            },
            error: function(error) {
                console.log(error)
            }
        });
    }, 6000)
  });


}).call(this);



var recognition = new webkitSpeechRecognition();
recognition.continuous = true;
recognition.interimResults = true;
recognition.lang = "fr-FR";
var isRecording = false;
recognition.onresult = function(event) {
  var interim_transcript = "";
  
  for (var i = event.resultIndex; i < event.results.length; ++i) {
    if (event.results[i].isFinal) {
      final_transcript += event.results[i][0].transcript;
    } else {
      interim_transcript += event.results[i][0].transcript;
    }
  }
  // Met à jour le texte affiché dans la page
 // document.getElementById("action-box-input-id").innerHTML = final_transcript;
  insertMessage(final_transcript);
  final_transcript = "";
};

var recordButton = document.getElementById('recordButton');
recordButton.addEventListener('click', function(){
  toggleRecording();
});

function toggleRecording() {
  if (!isRecording) {
    recognition.start();
    recordButton.style.backgroundColor = "#4CAF50";
    isRecording = true;
  } else {
    recognition.stop();
    recordButton.style.backgroundColor = "red";
    isRecording = false;
  }
}


window.addEventListener("load", function() {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();

      // Fonction pour démarrer la visualisation audio
      function startVisualization(stream) {
        // Créer un nœud de source de microphone
        const source = audioContext.createMediaStreamSource(stream);

        // Créer un nœud d'analyseur de niveau
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 128; // Définir la taille de la fenêtre de transformée de Fourier rapide (FFT)

        // Connecter le nœud de source au nœud d'analyseur
        source.connect(analyser);

        // Obtenir les données de niveau de l'analyseur de niveau
        const data = new Uint8Array(analyser.frequencyBinCount);
        var height;

        // Fonction de mise à jour des hauteurs des vagues
        function updateWaveform() {
          // Obtenir les niveaux d'intensités de la voix
          analyser.getByteFrequencyData(data);

          // Mettre à jour les hauteurs des vagues en fonction des niveaux d'intensités de la voix
          for (let i = 0; i < data.length; i++) {
            const wave = document.getElementsByClassName('wave')[i];
            height = data[i] / 2;
            wave.style.height = `${height}px`;
          }

          // Demander une nouvelle mise à jour des hauteurs des vagues
          requestAnimationFrame(updateWaveform);
        }

        // Démarrer la mise à jour des hauteurs des vagues
        updateWaveform();
      }

      // Obtenir l'accès au microphone
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(startVisualization)
        .catch((error) => {
          console.error('Erreur lors de lobtention de laccès au microphone:', error);
});});