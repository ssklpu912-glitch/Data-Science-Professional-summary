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

/* Password fields: adds a Show/Hide toggle to every password .form-input, and — where the
   field carries data-pw-rules — a live requirements checklist and strength bar. The rules
   here mirror validate_password() in app.py, which is what actually enforces them. */
document.addEventListener('DOMContentLoaded', function () {
	document.querySelectorAll('input[type="password"].form-input').forEach(function (input) {
		var wrap = document.createElement('div');
		wrap.className = 'pw-wrap';
		input.parentNode.insertBefore(wrap, input);
		wrap.appendChild(input);

		var toggle = document.createElement('button');
		toggle.type = 'button';
		toggle.className = 'pw-toggle';
		toggle.textContent = 'Show';
		toggle.setAttribute('aria-label', 'Show password');
		toggle.addEventListener('click', function () {
			var showing = input.type === 'text';
			input.type = showing ? 'password' : 'text';
			toggle.textContent = showing ? 'Show' : 'Hide';
			toggle.setAttribute('aria-label', showing ? 'Show password' : 'Hide password');
		});
		wrap.appendChild(toggle);

		if (input.hasAttribute('data-pw-rules')) {
			var rules = [
				['At least 8 characters', function (v) { return v.length >= 8; }],
				['At least one letter', function (v) { return /[A-Za-z]/.test(v); }],
				['At least one number', function (v) { return /[0-9]/.test(v); }]
			];
			var list = document.createElement('ul');
			list.className = 'pw-rules';
			var items = rules.map(function (r) {
				var li = document.createElement('li');
				li.textContent = r[0];
				list.appendChild(li);
				return li;
			});
			var meter = document.createElement('div');
			meter.className = 'pw-strength';
			var bar = document.createElement('span');
			meter.appendChild(bar);
			var note = document.createElement('p');
			note.className = 'pw-hint';
			note.textContent = 'Longer is stronger — a symbol, mixed case, or 12+ characters makes it harder to guess.';
			wrap.parentNode.appendChild(meter);
			wrap.parentNode.appendChild(list);
			wrap.parentNode.appendChild(note);

			input.addEventListener('input', function () {
				var v = input.value;
				var met = 0;
				rules.forEach(function (r, i) {
					var ok = r[1](v);
					items[i].classList.toggle('ok', ok);
					if (ok) met++;
				});
				var score = met;
				if (v.length >= 12) score++;
				if (/[^A-Za-z0-9]/.test(v)) score++;
				if (/[a-z]/.test(v) && /[A-Z]/.test(v)) score++;
				var pct = Math.min(score / 6, 1) * 100;
				bar.style.width = (v ? pct : 0) + '%';
				bar.style.background = pct < 45 ? '#c0392b' : (pct < 80 ? '#d4a017' : '#2a6f2a');
			});
		}
	});
});

/* Remembered emails: once someone is signed in, their email is kept in this browser only
   (localStorage — never sent anywhere) so the sign-in email box can offer it next time,
   alongside whatever the browser's own password manager suggests. The server deliberately
   never lists registered accounts, which would let anyone enumerate users. */
document.addEventListener('DOMContentLoaded', function () {
	var KEY = 'lablogbook.recentEmails';
	function load() {
		try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; }
	}
	var meta = document.querySelector('meta[name="signed-in-email"]');
	if (meta) {
		var emails = load().filter(function (e) { return e !== meta.content; });
		emails.unshift(meta.content);
		try { localStorage.setItem(KEY, JSON.stringify(emails.slice(0, 5))); } catch (e) {}
	}
	document.querySelectorAll('input[data-remembered-emails]').forEach(function (input) {
		var saved = load();
		if (!saved.length) return;
		var list = document.createElement('datalist');
		list.id = 'remembered-emails-' + Math.random().toString(36).slice(2);
		saved.forEach(function (e) {
			var o = document.createElement('option');
			o.value = e;
			list.appendChild(o);
		});
		input.setAttribute('list', list.id);
		input.parentNode.appendChild(list);
	});
});
