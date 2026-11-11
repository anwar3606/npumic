// NPU Denoised Mic: quick settings toggle for the npumic systemd user service.
//
// The toggle starts and stops the service (systemctl --user). Its menu shows the state, the microphone in use and any
// problems reported by the service, switches its processing steps, records a voice sample, and opens the debug page or
// the service log. A top bar icon appears while an app uses the microphone, and a warning icon while there is a problem;
// each new problem is also shown as a notification.
//
// The service is reached through its local HTTP API (see src/npumic/control.py). All I/O is asynchronous.
import GLib from 'gi://GLib';
import Gio from 'gi://Gio';
import GObject from 'gi://GObject';
import Soup from 'gi://Soup?version=3.0';
import {Extension} from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';
import {QuickMenuToggle, SystemIndicator} from 'resource:///org/gnome/shell/ui/quickSettings.js';

Gio._promisify(Soup.Session.prototype, 'send_and_read_async');
Gio._promisify(Gio.Subprocess.prototype, 'communicate_utf8_async');
Gio._promisify(Gio.File.prototype, 'replace_contents_async');

const API = 'http://127.0.0.1:5057';
const UNIT = 'npumic.service';
const POLL_SECONDS = 2;
const LOG_LINES = 400;
const MIC_ICON = 'audio-input-microphone-symbolic';
const WARNING_ICON = 'dialog-warning-symbolic';
const DEVICES = {NPU: 'on the NPU', CPU: 'on the CPU', none: 'failed, audio passes through'};

async function run(argv, cancellable) {
    const proc = Gio.Subprocess.new(argv, Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_PIPE);
    const [stdout, stderr] = await proc.communicate_utf8_async(null, cancellable);
    return {ok: proc.get_successful(), stdout: stdout ?? '', stderr: stderr ?? ''};
}

const MicToggle = GObject.registerClass(
class MicToggle extends QuickMenuToggle {
    _init(owner) {
        super._init({title: 'NPU Mic', iconName: MIC_ICON, toggleMode: false});
        this._owner = owner;
        this.menu.setHeader(MIC_ICON, 'NPU Denoised Mic', 'Checking…');

        this._stateItem = this._infoItem();
        this._micItem = this._infoItem();
        this._problemsSection = new PopupMenu.PopupMenuSection();
        this.menu.addMenuItem(this._problemsSection);
        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        this._noiseItem = this._switch('Noise removal', 'noise_removal');
        this._isolationItem = this._switch('Keep only my voice', 'isolation');
        this._levelItem = this._switch('Automatic mic level', 'auto_level');
        this._recordItem = this.menu.addAction('Record voice sample (talk alone for 10 s)',
            () => owner.post({record_voice: true}));
        this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());
        this.menu.addAction('Open debug page', () => owner.openDebugPage());
        this.menu.addAction('Show log', () => owner.showLog());

        this.connect('clicked', () => owner.toggleService());
    }

    _infoItem() {
        const item = new PopupMenu.PopupMenuItem('', {reactive: false});
        this.menu.addMenuItem(item);
        return item;
    }

    _switch(label, key) {
        const item = new PopupMenu.PopupSwitchMenuItem(label, false);
        item.connect('toggled', (_item, state) => this._owner.post({[key]: state}));
        this.menu.addMenuItem(item);
        return item;
    }

    _setProblems(problems) {
        this._problemsSection.removeAll();
        for (const problem of problems) {
            const item = new PopupMenu.PopupImageMenuItem(problem, WARNING_ICON, {reactive: false});
            item.label.clutter_text.line_wrap = true;
            this._problemsSection.addMenuItem(item);
        }
    }

    showStopped(serviceState) {
        const failed = serviceState === 'failed';
        this.checked = false;
        this.subtitle = failed ? 'Failed' : 'Off';
        this.menu.setHeader(failed ? WARNING_ICON : MIC_ICON, 'NPU Denoised Mic',
            failed ? 'The service stopped with an error; see the log' : 'Off: apps use the microphone directly');
        this._stateItem.label.text = serviceState === 'activating' ? 'Starting…' : 'Click the toggle to start';
        this._micItem.visible = false;
        this._setProblems([]);
        for (const item of [this._noiseItem, this._isolationItem, this._levelItem, this._recordItem])
            item.visible = false;
    }

    showStatus(s) {
        this.checked = true;
        const active = s.state === 'active';
        this.subtitle = s.problems.length ? 'Problem' : active ? 'In use' : 'Ready';
        const device = s.noise_removal ? `DPDFNet ${DEVICES[s.noise_removal_device] ?? s.noise_removal_device}` : 'noise removal off';
        this.menu.setHeader(s.problems.length ? WARNING_ICON : MIC_ICON, 'NPU Denoised Mic',
            active ? `In use by ${s.apps.join(', ') || 'no app'} · ${device}` : `Idle until an app uses it · ${device}`);
        this._stateItem.label.text = `Delay ${s.latency_ms} ms · ${s.xruns} glitches, ${s.late_cycles} late cycles since start`;
        this._micItem.visible = true;
        this._micItem.label.text = `Microphone: ${s.mic_label ?? 'none'}${s.mic_choice ? '' : ' (automatic)'}`;
        this._setProblems(s.problems);

        for (const item of [this._noiseItem, this._isolationItem, this._levelItem, this._recordItem])
            item.visible = true;
        this._noiseItem.setToggleState(s.noise_removal);
        this._levelItem.setToggleState(s.auto_level);
        this._isolationItem.setToggleState(s.isolation);
        this._isolationItem.setSensitive(s.voice.ready);
        this._isolationItem.label.text = s.voice.ready
            ? `Keep only my voice${s.isolation ? ` (${DEVICES[s.isolation_device] ?? s.isolation_device})` : ''}`
            : 'Keep only my voice (record a voice sample first)';
        const progress = s.voice.record_progress;
        this._recordItem.setSensitive(progress === null);
        this._recordItem.label.text = progress === null
            ? `${s.voice.ready ? 'Record voice sample again' : 'Record voice sample'} (talk alone for 10 s)`
            : progress < 1 ? `Recording… ${Math.ceil((1 - progress) * 10)} s left` : 'Preparing the sample…';
    }
});

const MicIndicator = GObject.registerClass(
class MicIndicator extends SystemIndicator {
    _init(owner) {
        super._init();
        this._icon = this._addIndicator();
        this._icon.visible = false;
        this.toggle = new MicToggle(owner);
        this.quickSettingsItems.push(this.toggle);
    }

    show(iconName) {
        this._icon.visible = iconName !== null;
        if (iconName)
            this._icon.icon_name = iconName;
    }
});

export default class NpuMicExtension extends Extension {
    enable() {
        this._session = new Soup.Session({timeout: 3});
        this._cancellable = new Gio.Cancellable();
        this._busy = false;
        this._notified = new Set();
        this._indicator = new MicIndicator(this);
        Main.panel.statusArea.quickSettings.addExternalIndicator(this._indicator);
        this._timer = GLib.timeout_add_seconds(GLib.PRIORITY_DEFAULT, POLL_SECONDS, () => {
            this.refresh();
            return GLib.SOURCE_CONTINUE;
        });
        this.refresh();
    }

    disable() {
        if (this._timer) {
            GLib.source_remove(this._timer);
            this._timer = 0;
        }
        this._cancellable?.cancel();
        this._cancellable = null;
        this._indicator?.quickSettingsItems.forEach(item => item.destroy());
        this._indicator?.destroy();
        this._indicator = null;
        this._session?.abort();
        this._session = null;
        this._notified = null;
    }

    async _request(method, path, body = null) {
        const message = Soup.Message.new(method, `${API}${path}`);
        if (body !== null)
            message.set_request_body_from_bytes('application/json', new GLib.Bytes(new TextEncoder().encode(JSON.stringify(body))));
        const bytes = await this._session.send_and_read_async(message, GLib.PRIORITY_DEFAULT, this._cancellable);
        const data = JSON.parse(new TextDecoder().decode(bytes.get_data()));
        if (message.get_status() !== Soup.Status.OK)
            throw new Error(data?.error ?? `HTTP ${message.get_status()}`);
        return data;
    }

    async refresh() {
        if (this._busy || !this._indicator)
            return;
        this._busy = true;
        try {
            const status = await this._request('GET', '/api/status');
            this._show(status);
        } catch (e) {
            if (!this._cancellable || this._cancellable.is_cancelled())
                return;
            await this._showStopped();
        } finally {
            this._busy = false;
        }
    }

    _show(status) {
        if (!this._indicator)
            return;
        this._indicator.toggle.showStatus(status);
        this._indicator.show(status.problems.length ? WARNING_ICON : status.state === 'active' ? MIC_ICON : null);
        for (const problem of status.problems) {
            if (!this._notified.has(problem)) {
                this._notified.add(problem);
                Main.notify('NPU Denoised Mic', problem);
            }
        }
        for (const problem of [...this._notified]) {
            if (!status.problems.includes(problem))
                this._notified.delete(problem);
        }
    }

    async _showStopped() {
        let state = 'unknown';
        try {
            state = (await run(['systemctl', '--user', 'is-active', UNIT], this._cancellable)).stdout.trim();
        } catch (e) {
            console.warn(`npumic: systemctl failed: ${e.message}`);
        }
        if (!this._indicator)
            return;
        this._indicator.toggle.showStopped(state);
        this._indicator.show(state === 'failed' ? WARNING_ICON : null);
        if (state === 'failed' && !this._notified.has('service-failed')) {
            this._notified.add('service-failed');
            Main.notify('NPU Denoised Mic', 'The service stopped with an error. Open the menu and choose Show log.');
        } else if (state !== 'failed') {
            this._notified.delete('service-failed');
        }
    }

    async post(settings) {
        try {
            this._show(await this._request('POST', '/api/settings', settings));
        } catch (e) {
            if (this._cancellable && !this._cancellable.is_cancelled())
                Main.notify('NPU Denoised Mic', e.message);
            this.refresh();
        }
    }

    async toggleService() {
        const running = this._indicator?.toggle.checked;
        try {
            const result = await run(['systemctl', '--user', running ? 'stop' : 'start', UNIT], this._cancellable);
            if (!result.ok)
                Main.notify('NPU Denoised Mic', result.stderr.trim() || `Could not ${running ? 'stop' : 'start'} the service`);
        } catch (e) {
            console.warn(`npumic: systemctl failed: ${e.message}`);
        }
        this.refresh();
    }

    openDebugPage() {
        try {
            Gio.AppInfo.launch_default_for_uri(`${API}/`, null);
        } catch (e) {
            Main.notify('NPU Denoised Mic', `Could not open the debug page: ${e.message}`);
        }
    }

    async showLog() {
        try {
            const result = await run(['journalctl', '--user', '-u', UNIT, '-n', String(LOG_LINES), '--no-pager',
                '-o', 'short-iso'], this._cancellable);
            const file = Gio.File.new_for_path(GLib.build_filenamev([GLib.get_user_cache_dir(), 'npumic.log']));
            const text = result.stdout || result.stderr || 'The log is empty.';
            await file.replace_contents_async(new TextEncoder().encode(text), null, false,
                Gio.FileCreateFlags.REPLACE_DESTINATION, this._cancellable);
            Gio.AppInfo.launch_default_for_uri(file.get_uri(), null);
        } catch (e) {
            if (this._cancellable && !this._cancellable.is_cancelled())
                Main.notify('NPU Denoised Mic', `Could not show the log: ${e.message}`);
        }
    }
}
