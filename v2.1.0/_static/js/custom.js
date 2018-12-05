$(document).ready(function() {

  // Dropdown menu
  var triggers = document.querySelectorAll(
    '.main-nav li.menu-item-has-children'
  );
  var background = document.querySelector('.dropdownBackground');
  var nav = document.querySelector('.main-nav');

  function handleEnter() {
    this.classList.add('trigger-enter');
    setTimeout(
      () =>
        this.classList.contains('trigger-enter') &&
        this.classList.add('trigger-enter-active'),
      150
    );
    background.classList.add('open');

    var dropdown = this.querySelector('.main-nav .sub-menu');
    var dropdownCoords = dropdown.getBoundingClientRect();
    var navCoords = nav.getBoundingClientRect();

    var coords = {
      height: dropdownCoords.height,
      width: dropdownCoords.width,
      top: dropdownCoords.top - navCoords.top + 5,
      left: dropdownCoords.left - navCoords.left
    };

    background.style.setProperty('width', `${coords.width}px`);
    background.style.setProperty('height', `${coords.height}px`);
    background.style.setProperty(
      'transform',
      `translate(${coords.left}px, ${coords.top}px)`
    );
  }

  function handleLeave() {
    this.classList.remove('trigger-enter', 'trigger-enter-active');
    background.classList.remove('open');
  }

  triggers.forEach(trigger =>
                   trigger.addEventListener('mouseenter', handleEnter)
                  );
  triggers.forEach(trigger =>
                   trigger.addEventListener('mouseleave', handleLeave)
                  );

  // Mobile slide out menus
  $('#mobile-nav').mmenu({
    slidingSubmenus: false,
    offCanvas: {
      position: 'right'
    }
  });
  var MobileNavAPI = $('#mobile-nav').data('mmenu');
  $('#mobile-close-button').click(function() {
    MobileNavAPI.close();
  });

  // Set height on sidebar
  var viewportHeight = Math.max(document.documentElement.clientHeight, window.innerHeight || 0);
  var $sidebar = $('.bs-docs-sidebar');
  $sidebar.css('max-height', `${viewportHeight - 40}px`);

  // Set up affix and scrollspy
  var $navAffix = $('#nav-affix');
  var $contentEl = $('.wrapper-content-right');

  // Desktop only
  if( !$('#mobile-menu-toggler').is(':visible') ) {
    // Only if the content is taller than the sidebar,
    // to avoid scroll jerk
    if ( $contentEl.height() > $navAffix.height() ) {
      // Init
      $navAffix.affix({
        offset: {
          top: 100,
          bottom: function () {
            return (this.bottom = $('footer').outerHeight(true) + 200)
          }
        }
      });
      // Recalculate the sidebar
      $(window).resize(function() {
        $navAffix.affix('checkPosition');
      });
    }
    $('body').scrollspy();
  } else {
    $('.wrapper-navleft').hide();
    $('#mobile-menu-toggler').click(function() {
      $('.wrapper-navleft').toggle();
    });
  }
});
