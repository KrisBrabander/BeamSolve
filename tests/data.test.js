const profileData = require('../js/data.js');

describe('data calculations', () => {
  beforeEach(() => {
    profileData.profiles = {
      TEST: [
        { name: 'T1', weightPerMeter: 10, surfaceAreaPerMeter: 2 }
      ]
    };
  });

  test('calculateWeight handles mm to m conversion', () => {
    const wMm = profileData.calculateWeight('TEST', 'T1', 1000, 2, 'mm');
    const wM = profileData.calculateWeight('TEST', 'T1', 1, 2, 'm');
    expect(wMm).toBeCloseTo(wM, 6);
  });

  test('calculateWeight handles inch and ft', () => {
    const wInch = profileData.calculateWeight('TEST', 'T1', 39.3701, 1, 'inch');
    const wFt = profileData.calculateWeight('TEST', 'T1', 3.28084, 1, 'ft');
    const expected = profileData.calculateWeight('TEST', 'T1', 1, 1, 'm');
    expect(wInch).toBeCloseTo(expected, 4);
    expect(wFt).toBeCloseTo(expected, 4);
  });

  test('calculateVolume uses weight and density', () => {
    const volume = profileData.calculateVolume('TEST', 'T1', 1, 2, 'm', 8000);
    const expectedWeight = profileData.calculateWeight('TEST', 'T1', 1, 2, 'm');
    expect(volume).toBeCloseTo(expectedWeight / 8000, 6);
  });
});
