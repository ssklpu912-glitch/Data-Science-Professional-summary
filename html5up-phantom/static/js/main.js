/*
	Phantom by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function($) {

	var	$window = $(window),
		$body = $('body');

	// Breakpoints.
		breakpoints({
			xlarge:   [ '1281px',  '1680px' ],
			large:    [ '981px',   '1280px' ],
			medium:   [ '737px',   '980px'  ],
			small:    [ '481px',   '736px'  ],
			xsmall:   [ '361px',   '480px'  ],
			xxsmall:  [ null,      '360px'  ]
		});

	// Play initial animations on page load.
		$window.on('load', function() {
			window.setTimeout(function() {
				$body.removeClass('is-preload');
			}, 100);
		});

	// Touch?
		if (browser.mobile)
			$body.addClass('is-touch');

	// Forms.
		var $form = $('form');

		// Auto-resizing textareas.
			$form.find('textarea').each(function() {

				var $this = $(this),
					$wrapper = $('<div class="textarea-wrapper"></div>'),
					$submits = $this.find('input[type="submit"]');

				$this
					.wrap($wrapper)
					.attr('rows', 1)
					.css('overflow', 'hidden')
					.css('resize', 'none')
					.on('keydown', function(event) {

						if (event.keyCode == 13
						&&	event.ctrlKey) {

							event.preventDefault();
							event.stopPropagation();

							$(this).blur();

						}

					})
					.on('blur focus', function() {
						$this.val($.trim($this.val()));
					})
					.on('input blur focus --init', function() {

						$wrapper
							.css('height', $this.height());

						$this
							.css('height', 'auto')
							.css('height', $this.prop('scrollHeight') + 'px');

					})
					.on('keyup', function(event) {

						if (event.keyCode == 9)
							$this
								.select();

					})
					.triggerHandler('--init');

				// Fix.
					if (browser.name == 'ie'
					||	browser.mobile)
						$this
							.css('max-height', '10em')
							.css('overflow-y', 'auto');

			});

	// Menu.
		var $menu = $('#menu');

		$menu.wrapInner('<div class="inner"></div>');

		$menu._locked = false;

		$menu._lock = function() {

			if ($menu._locked)
				return false;

			$menu._locked = true;

			window.setTimeout(function() {
				$menu._locked = false;
			}, 350);

			return true;

		};

		$menu._show = function() {

			if ($menu._lock())
				$body.addClass('is-menu-visible');

		};

		$menu._hide = function() {

			if ($menu._lock())
				$body.removeClass('is-menu-visible');

		};

		$menu._toggle = function() {

			if ($menu._lock())
				$body.toggleClass('is-menu-visible');

		};

		$menu
			.appendTo($body)
			.on('click', function(event) {
				event.stopPropagation();
			})
			.on('click', 'a', function(event) {

				var href = $(this).attr('href');

				event.preventDefault();
				event.stopPropagation();

				// Hide.
					$menu._hide();

				// Redirect.
					if (href == '#menu')
						return;

					window.setTimeout(function() {
						window.location.href = href;
					}, 350);

			})
			.append('<a class="close" href="#menu">Close</a>');

		$body
			.on('click', 'a[href="#menu"]', function(event) {

				event.stopPropagation();
				event.preventDefault();

				// Toggle.
					$menu._toggle();

			})
			.on('click', function(event) {

				// Hide.
					$menu._hide();

			})
			.on('keydown', function(event) {

				// Hide on escape.
					if (event.keyCode == 27)
						$menu._hide();

			});

})(jQuery);
/* Lightweight replacement for window.confirm(), which some embedded/sandboxed
   preview browsers suppress (silently returning false, so the destructive
   action never fires). Turns the trigger element into an inline
   "Click again to confirm" prompt for a few seconds instead of relying on a
   native dialog. Usage:
     - onsubmit="return appConfirm(this.querySelector('[type=submit],button'), 'Delete this?');"
     - onclick="if (!appConfirm(this, 'Delete this?')) return; doTheThing();"
*/
function appConfirm(el, message) {
	if (!el) return window.confirm(message);
	if (el.dataset.confirmArmed === '1') {
		delete el.dataset.confirmArmed;
		if (el._appConfirmTimeout) clearTimeout(el._appConfirmTimeout);
		if (el._appConfirmOriginal !== undefined) {
			if ('value' in el) el.value = el._appConfirmOriginal;
			else el.textContent = el._appConfirmOriginal;
		}
		el.style.color = '';
		return true;
	}
	el.dataset.confirmArmed = '1';
	el._appConfirmOriginal = ('value' in el && el.tagName !== 'BUTTON') ? el.value : el.textContent;
	var prompt = 'Click again to confirm';
	if ('value' in el && el.tagName !== 'BUTTON') el.value = prompt;
	else el.textContent = prompt;
	el.style.color = '#b3261e';
	el._appConfirmTimeout = setTimeout(function () {
		delete el.dataset.confirmArmed;
		if ('value' in el && el.tagName !== 'BUTTON') el.value = el._appConfirmOriginal;
		else el.textContent = el._appConfirmOriginal;
		el.style.color = '';
	}, 5000);
	return false;
}

/* CSRF protection: Flask-WTF's CSRFProtect checks every POST/PUT/PATCH/DELETE
   request for a 'csrf_token' field matching the session. The token itself is
   published once per page load via <meta name="csrf-token">. */
function getCsrfToken() {
	var meta = document.querySelector('meta[name="csrf-token"]');
	return meta ? meta.getAttribute('content') : '';
}

function addCsrfToken(form) {
	if (form.querySelector('input[name="csrf_token"]')) return;
	var input = document.createElement('input');
	input.type = 'hidden';
	input.name = 'csrf_token';
	input.value = getCsrfToken();
	form.appendChild(input);
}

document.addEventListener('DOMContentLoaded', function () {
	var forms = document.querySelectorAll('form');
	for (var i = 0; i < forms.length; i++) {
		var method = (forms[i].getAttribute('method') || 'GET').toUpperCase();
		if (method === 'POST') addCsrfToken(forms[i]);
	}
});
