import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import MainTabNavigator from './MainTabNavigator';
import { DetectionScreen, GeminiNanoTestScreen, EntryDetailScreen, FoodSearchScreen, BarcodeScanScreen, RecipeScreen, QuickAddScreen, ReidentifyMergeScreen, GalleryScanScreen, ScaleInputScreen, WeightTrendScreen } from '../screens';
import SyncSettingsScreen from '../screens/SyncSettingsScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

/** Temporary placeholder until Plan 03 creates AddFoodScreen */
function AddFoodPlaceholder() {
  return (
    <View style={addFoodStyles.container}>
      <Text style={addFoodStyles.text}>Add Food - Coming in Plan 03</Text>
    </View>
  );
}

const addFoodStyles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#F5F5F5' },
  text: { fontSize: 16, color: '#6B7280', fontWeight: '600' },
});

export default function RootNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
        }}
      >
        <Stack.Screen name="Main" component={MainTabNavigator} />
        <Stack.Screen
          name="AddFood"
          component={AddFoodPlaceholder}
          options={{
            animation: 'slide_from_right',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="Detection"
          component={DetectionScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
          }}
        />
        <Stack.Screen
          name="GeminiNanoTest"
          component={GeminiNanoTestScreen}
          options={{
            headerShown: true,
            headerTitle: 'Gemini Nano Test',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="FoodSearch"
          component={FoodSearchScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="BarcodeScan"
          component={BarcodeScanScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="Recipes"
          component={RecipeScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="QuickAdd"
          component={QuickAddScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="EntryDetail"
          component={EntryDetailScreen}
          options={{
            headerShown: true,
            headerTitle: 'Meal Detail',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="SyncSettings"
          component={SyncSettingsScreen}
          options={{
            headerShown: true,
            headerTitle: 'Google Drive Sync',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="ReidentifyMerge"
          component={ReidentifyMergeScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="GalleryScan"
          component={GalleryScanScreen}
          options={{
            headerShown: true,
            headerTitle: 'Gallery Scan',
            animation: 'slide_from_right',
          }}
        />
        <Stack.Screen
          name="ScaleInput"
          component={ScaleInputScreen}
          options={{
            animation: 'slide_from_bottom',
            presentation: 'fullScreenModal',
            headerShown: false,
          }}
        />
        <Stack.Screen
          name="WeightTrend"
          component={WeightTrendScreen}
          options={{
            animation: 'slide_from_right',
            headerShown: false,
          }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
