/**
 * SyncSettingsScreen -- Google Drive sync management.
 *
 * Google account sign-in/out, WiFi-only toggle, auto-resolve toggle,
 * manual sync, full backup upload, restore from Drive, conflict review,
 * and custom Drive folder option.
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  Switch,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useSyncStore } from '../store/useSyncStore';
import { signInToGoogle, signOutGoogle, addDriveFileScope } from '../services/sync/driveAuth';
import { triggerManualSync } from '../services/sync/syncScheduler';
import { performFullBackup } from '../services/backup/backupService';
import { uploadFullBackup } from '../services/sync/driveSync';
import { discoverRemoteBackups, restoreFromDrive } from '../services/sync/restoreService';
import ConflictResolverModal from '../components/sync/ConflictResolverModal';
import { Paths } from 'expo-file-system';

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function SyncSettingsScreen() {
  const {
    signedIn,
    userEmail,
    lastSyncAt,
    syncStatus,
    pendingConflicts,
    wifiOnly,
    autoResolve,
    setSignedIn,
    setWifiOnly,
    setAutoResolve,
    clearSync,
  } = useSyncStore();

  const [syncing, setSyncing] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [conflictModalVisible, setConflictModalVisible] = useState(false);
  const [customFolder, setCustomFolder] = useState(false);

  // -----------------------------------------------------------------------
  // Google account
  // -----------------------------------------------------------------------

  async function handleSignIn() {
    try {
      const result = await signInToGoogle();
      const email =
        (result as { data?: { user?: { email?: string } } })?.data?.user?.email ??
        (result as { user?: { email?: string } })?.user?.email ??
        null;
      setSignedIn(true, email);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Sign-in failed';
      Alert.alert('Sign-in Error', msg);
    }
  }

  async function handleSignOut() {
    try {
      await signOutGoogle();
      clearSync();
    } catch {
      Alert.alert('Error', 'Failed to sign out.');
    }
  }

  // -----------------------------------------------------------------------
  // Manual actions
  // -----------------------------------------------------------------------

  async function handleSyncNow() {
    setSyncing(true);
    try {
      await triggerManualSync();
      Alert.alert('Sync Complete', 'Your data has been synced to Google Drive.');
    } catch {
      Alert.alert('Sync Failed', 'Unable to sync. Please try again.');
    } finally {
      setSyncing(false);
    }
  }

  async function handleUploadFull() {
    setUploading(true);
    try {
      const result = await performFullBackup();
      const localPath = `${Paths.document.uri}/backups/${result.filename}`;
      await uploadFullBackup(result.filename, localPath);
      Alert.alert('Upload Complete', `Full backup uploaded: ${result.filename}`);
    } catch {
      Alert.alert('Upload Failed', 'Unable to upload full backup.');
    } finally {
      setUploading(false);
    }
  }

  async function handleRestore() {
    setRestoring(true);
    try {
      const manifest = await discoverRemoteBackups();
      if (!manifest) {
        Alert.alert('No Backups', 'No remote backups found on Google Drive.');
        setRestoring(false);
        return;
      }

      const lastSync = manifest.lastSyncedAt
        ? new Date(manifest.lastSyncedAt).toLocaleString()
        : 'Unknown';

      Alert.alert(
        'Restore from Drive',
        `Found backup from ${lastSync}.\n\nThis will replace your local data. The app will need to restart after restore.`,
        [
          { text: 'Cancel', style: 'cancel', onPress: () => setRestoring(false) },
          {
            text: 'Restore',
            style: 'destructive',
            onPress: async () => {
              try {
                await restoreFromDrive();
                Alert.alert('Restore Complete', 'Data restored. Please restart the app.');
              } catch (err) {
                const msg = err instanceof Error ? err.message : 'Restore failed';
                Alert.alert('Restore Failed', msg);
              } finally {
                setRestoring(false);
              }
            },
          },
        ],
      );
    } catch {
      Alert.alert('Error', 'Failed to check for remote backups.');
      setRestoring(false);
    }
  }

  // -----------------------------------------------------------------------
  // Custom folder scope escalation
  // -----------------------------------------------------------------------

  async function handleCustomFolderToggle(enabled: boolean) {
    if (enabled) {
      try {
        await addDriveFileScope();
        setCustomFolder(true);
      } catch {
        Alert.alert('Error', 'Failed to request Drive file access.');
      }
    } else {
      setCustomFolder(false);
    }
  }

  // -----------------------------------------------------------------------
  // Status helpers
  // -----------------------------------------------------------------------

  function statusColor(): string {
    switch (syncStatus) {
      case 'syncing':
        return '#F59E0B';
      case 'error':
        return '#EF4444';
      case 'conflict':
        return '#F59E0B';
      default:
        return '#16A34A';
    }
  }

  function statusLabel(): string {
    switch (syncStatus) {
      case 'syncing':
        return 'Syncing...';
      case 'error':
        return 'Error';
      case 'conflict':
        return 'Conflicts';
      default:
        return 'Up to date';
    }
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Google Drive Sync</Text>

      {/* Google Account */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Google Account</Text>
        {signedIn ? (
          <>
            <View style={styles.row}>
              <Ionicons name="person-circle-outline" size={20} color="#16A34A" />
              <Text style={styles.emailText}>{userEmail ?? 'Signed in'}</Text>
            </View>
            <Pressable style={styles.actionBtn} onPress={handleSignOut}>
              <Ionicons name="log-out-outline" size={18} color="#EF4444" />
              <Text style={[styles.actionBtnText, { color: '#EF4444' }]}>Sign Out</Text>
            </Pressable>
          </>
        ) : (
          <Pressable style={[styles.actionBtn, styles.primaryBtn]} onPress={handleSignIn}>
            <Ionicons name="logo-google" size={18} color="#FFF" />
            <Text style={[styles.actionBtnText, { color: '#FFF' }]}>Sign in with Google</Text>
          </Pressable>
        )}
      </View>

      {/* Status */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Sync Status</Text>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>Status</Text>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
            <View style={[styles.statusDot, { backgroundColor: statusColor() }]} />
            <Text style={styles.rowValue}>{statusLabel()}</Text>
          </View>
        </View>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>Last synced</Text>
          <Text style={styles.rowValue}>
            {lastSyncAt ? relativeTime(lastSyncAt) : 'Never'}
          </Text>
        </View>
      </View>

      {/* Sync Preferences */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Sync Preferences</Text>
        <View style={styles.toggleRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>WiFi only</Text>
            <Text style={styles.hint}>Only sync when connected to WiFi</Text>
          </View>
          <Switch
            value={wifiOnly}
            onValueChange={setWifiOnly}
            trackColor={{ true: '#16A34A' }}
          />
        </View>
        <View style={styles.toggleRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>Auto-resolve conflicts</Text>
            <Text style={styles.hint}>
              When enabled, conflicts are automatically resolved using the most recent change
            </Text>
          </View>
          <Switch
            value={autoResolve}
            onValueChange={setAutoResolve}
            trackColor={{ true: '#16A34A' }}
          />
        </View>
      </View>

      {/* Manual Actions */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Manual Actions</Text>
        {syncing || uploading || restoring ? (
          <ActivityIndicator size="small" color="#16A34A" style={{ paddingVertical: 16 }} />
        ) : (
          <>
            <Pressable style={styles.actionBtn} onPress={handleSyncNow} disabled={!signedIn}>
              <Ionicons name="sync-outline" size={18} color={signedIn ? '#16A34A' : '#D1D5DB'} />
              <Text style={[styles.actionBtnText, !signedIn && { color: '#D1D5DB' }]}>
                Sync Now
              </Text>
            </Pressable>
            <Pressable style={styles.actionBtn} onPress={handleUploadFull} disabled={!signedIn}>
              <Ionicons name="cloud-upload-outline" size={18} color={signedIn ? '#3B82F6' : '#D1D5DB'} />
              <Text style={[styles.actionBtnText, { color: signedIn ? '#3B82F6' : '#D1D5DB' }]}>
                Upload Full Backup
              </Text>
            </Pressable>
            <Pressable style={styles.actionBtn} onPress={handleRestore} disabled={!signedIn}>
              <Ionicons name="cloud-download-outline" size={18} color={signedIn ? '#7C3AED' : '#D1D5DB'} />
              <Text style={[styles.actionBtnText, { color: signedIn ? '#7C3AED' : '#D1D5DB' }]}>
                Restore from Drive
              </Text>
            </Pressable>
          </>
        )}
      </View>

      {/* Conflict Review */}
      {pendingConflicts.length > 0 && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Conflicts</Text>
          <View style={styles.row}>
            <View style={styles.conflictBadge}>
              <Text style={styles.conflictBadgeText}>
                {pendingConflicts.length} unresolved conflict{pendingConflicts.length !== 1 ? 's' : ''}
              </Text>
            </View>
          </View>
          <Pressable
            style={[styles.actionBtn, { borderColor: '#F59E0B' }]}
            onPress={() => setConflictModalVisible(true)}
          >
            <Ionicons name="warning-outline" size={18} color="#F59E0B" />
            <Text style={[styles.actionBtnText, { color: '#F59E0B' }]}>Review Conflicts</Text>
          </Pressable>
        </View>
      )}

      {/* Custom Folder */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Advanced</Text>
        <View style={styles.toggleRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.rowLabel}>Use custom Drive folder</Text>
            <Text style={styles.hint}>
              Default: hidden app data folder. Enable to use a visible folder.
            </Text>
          </View>
          <Switch
            value={customFolder}
            onValueChange={handleCustomFolderToggle}
            trackColor={{ true: '#7C3AED' }}
            disabled={!signedIn}
          />
        </View>
      </View>

      <View style={{ height: 100 }} />

      <ConflictResolverModal
        visible={conflictModalVisible}
        onClose={() => setConflictModalVisible(false)}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F5F5' },
  content: { paddingTop: 60, paddingHorizontal: 16 },
  title: { fontSize: 28, fontWeight: '800', color: '#111827', marginBottom: 20 },

  card: {
    backgroundColor: '#FFF',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 3,
  },
  cardTitle: { fontSize: 18, fontWeight: '700', color: '#111827', marginBottom: 12 },

  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    gap: 8,
  },
  rowLabel: { fontSize: 15, color: '#374151' },
  rowValue: { fontSize: 15, color: '#6B7280' },

  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#F3F4F6',
  },

  hint: { fontSize: 12, color: '#9CA3AF', marginTop: 2 },

  emailText: { fontSize: 15, color: '#374151', marginLeft: 8 },

  actionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#F3F4F6',
  },
  actionBtnText: { fontSize: 15, fontWeight: '500', color: '#16A34A' },

  primaryBtn: {
    backgroundColor: '#4285F4',
    borderRadius: 12,
    paddingHorizontal: 16,
    justifyContent: 'center',
    borderBottomWidth: 0,
  },

  statusDot: { width: 8, height: 8, borderRadius: 4 },

  conflictBadge: {
    backgroundColor: '#FEF3C7',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  conflictBadgeText: { fontSize: 13, fontWeight: '600', color: '#92400E' },
});
